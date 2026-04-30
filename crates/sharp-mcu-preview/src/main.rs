use std::{
    fs,
    path::PathBuf,
    sync::{
        Arc,
        mpsc::{self, Receiver, TryRecvError},
    },
    thread,
};

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use clap::Parser;
use image::ImageReader;
use sharp_mcu_core::{LayerManifest, PackageManifest, compute_layer_transform, stage_rotation_deg};
use winit::{
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::Window,
};

const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.045,
    g: 0.055,
    b: 0.085,
    a: 1.0,
};

const TEXTURED_SHADER: &str = r#"
struct VsIn {
  @location(0) pos: vec2<f32>,
  @location(1) uv: vec2<f32>,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
  var out: VsOut;
  out.pos = vec4<f32>(in.pos, 0.0, 1.0);
  out.uv = in.uv;
  return out;
}

@group(0) @binding(0) var layer_tex: texture_2d<f32>;
@group(0) @binding(1) var layer_sampler: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  // Layer PNGs are written with premultiplied alpha by sharp-mcu-core, so we
  // pass the sampled value through unchanged. Re-multiplying here would dim
  // alpha-edge texels twice; sampling straight alpha here would let bilinear
  // filtering paint dark contour-tracing halos along every layer boundary.
  return textureSample(layer_tex, layer_sampler, in.uv);
}
"#;

const COLOR_SHADER: &str = r#"
struct VsIn {
  @location(0) pos: vec2<f32>,
  @location(1) color: vec4<f32>,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
  var out: VsOut;
  out.pos = vec4<f32>(in.pos, 0.0, 1.0);
  out.color = in.color;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  return in.color;
}
"#;

#[derive(Parser)]
#[command(name = "sharp-mcu-preview")]
#[command(about = "Preview an exported MCU slice package with drag-driven virtual gyro")]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    tilt_deg: Option<f32>,
    #[arg(long)]
    parallax: Option<f32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let manifest_path = args
        .manifest
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", args.manifest.display()))?;
    let manifest_dir = manifest_path
        .parent()
        .context("manifest path had no parent directory")?;
    let manifest: PackageManifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", manifest_path.display()))?;

    let title = format!(
        "sharp-mcu preview · {}x{} · {} layers",
        manifest.package.width, manifest.package.height, manifest.package.layer_count
    );
    let layers = manifest
        .layers
        .iter()
        .map(|layer| LoadedLayer {
            meta: layer.clone(),
            path: manifest_dir.join(&layer.file),
            texture: None,
        })
        .collect::<Vec<_>>();

    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        event_loop.create_window(
            Window::default_attributes()
                .with_title(&title)
                .with_inner_size(LogicalSize::new(1440.0, 900.0))
                .with_min_inner_size(LogicalSize::new(960.0, 640.0)),
        )?,
    );
    let mut renderer = pollster::block_on(GpuRenderer::new(window.clone()))?;
    let mut state = PreviewState::new(
        manifest,
        layers,
        args.tilt_deg,
        args.parallax,
        manifest_dir.to_path_buf(),
        title,
    );
    window.set_title(&state.window_title());
    let loader_rx = spawn_layer_loader(
        state
            .layers
            .iter()
            .map(|layer| layer.path.clone())
            .collect(),
    );

    window.request_redraw();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Wait);
        match event {
            Event::WindowEvent { window_id, event } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(size) => {
                    if let Err(error) = renderer.resize(size) {
                        eprintln!("{error:#}");
                        elwt.exit();
                    } else {
                        window.request_redraw();
                    }
                }
                WindowEvent::RedrawRequested => {
                    if let Err(error) = renderer.render(&state) {
                        eprintln!("{error:#}");
                        elwt.exit();
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state == ElementState::Pressed && !event.repeat {
                        let mut changed = false;
                        match &event.logical_key {
                            Key::Character(ch) => match ch.as_str() {
                                "[" => {
                                    state.tilt_deg = (state.tilt_deg - 1.0).clamp(0.0, 60.0);
                                    changed = true;
                                }
                                "]" => {
                                    state.tilt_deg = (state.tilt_deg + 1.0).clamp(0.0, 60.0);
                                    changed = true;
                                }
                                "-" => {
                                    state.parallax = (state.parallax - 0.01).clamp(0.0, 1.0);
                                    changed = true;
                                }
                                "=" | "+" => {
                                    state.parallax = (state.parallax + 0.01).clamp(0.0, 1.0);
                                    changed = true;
                                }
                                "0" => {
                                    state.reset_motion();
                                    changed = true;
                                }
                                _ => {}
                            },
                            Key::Named(NamedKey::Escape) => {
                                state.reset_motion();
                                changed = true;
                            }
                            _ => {}
                        }
                        if changed {
                            window.set_title(&state.window_title());
                            elwt.set_control_flow(ControlFlow::Poll);
                            window.request_redraw();
                        }
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    state.cursor = Some(position);
                    if state.drag_active {
                        let rect =
                            device_pad_rect(window.inner_size().width, window.inner_size().height);
                        state.set_target_from_pointer(position, rect);
                        elwt.set_control_flow(ControlFlow::Poll);
                        window.request_redraw();
                    }
                }
                WindowEvent::MouseInput {
                    state: button_state,
                    button,
                    ..
                } if button == MouseButton::Left => {
                    if button_state == ElementState::Pressed {
                        if let Some(cursor) = state.cursor {
                            let rect = device_pad_rect(
                                window.inner_size().width,
                                window.inner_size().height,
                            );
                            if rect.contains(cursor) {
                                state.drag_active = true;
                                state.set_target_from_pointer(cursor, rect);
                                elwt.set_control_flow(ControlFlow::Poll);
                                window.request_redraw();
                            }
                        }
                    } else {
                        state.drag_active = false;
                        state.target_x = 0.0;
                        state.target_y = 0.0;
                        elwt.set_control_flow(ControlFlow::Poll);
                        window.request_redraw();
                    }
                }
                _ => {}
            },
            Event::AboutToWait => {
                let loaded = match integrate_loaded_layers(&loader_rx, &mut state, &renderer) {
                    Ok(changed) => changed,
                    Err(error) => {
                        eprintln!("{error:#}");
                        elwt.exit();
                        false
                    }
                };
                if state.tick() || loaded {
                    elwt.set_control_flow(ControlFlow::Poll);
                    window.request_redraw();
                }
            }
            _ => {}
        }
    })?;
    Ok(())
}

struct LoadedLayer {
    meta: LayerManifest,
    path: PathBuf,
    texture: Option<LayerTexture>,
}

struct PreviewState {
    manifest: PackageManifest,
    layers: Vec<LoadedLayer>,
    cursor: Option<PhysicalPosition<f64>>,
    drag_active: bool,
    x: f32,
    y: f32,
    target_x: f32,
    target_y: f32,
    tilt_deg: f32,
    parallax: f32,
    title_prefix: String,
    _manifest_dir: PathBuf,
}

impl PreviewState {
    fn new(
        manifest: PackageManifest,
        layers: Vec<LoadedLayer>,
        tilt_deg: Option<f32>,
        parallax: Option<f32>,
        manifest_dir: PathBuf,
        title_prefix: String,
    ) -> Self {
        let tilt_deg = tilt_deg.unwrap_or(manifest.motion.tilt_default_deg);
        let parallax = parallax.unwrap_or(manifest.motion.parallax_default);
        Self {
            manifest,
            layers,
            cursor: None,
            drag_active: false,
            x: 0.0,
            y: 0.0,
            target_x: 0.0,
            target_y: 0.0,
            tilt_deg,
            parallax,
            title_prefix,
            _manifest_dir: manifest_dir,
        }
    }

    fn tick(&mut self) -> bool {
        let prev_x = self.x;
        let prev_y = self.y;
        let stiffness = if self.drag_active { 0.35 } else { 0.18 };
        self.x += (self.target_x - self.x) * stiffness;
        self.y += (self.target_y - self.y) * stiffness;
        if (self.target_x - self.x).abs() < 0.0005 {
            self.x = self.target_x;
        }
        if (self.target_y - self.y).abs() < 0.0005 {
            self.y = self.target_y;
        }
        (self.x - prev_x).abs() > 0.0001 || (self.y - prev_y).abs() > 0.0001
    }

    fn set_target_from_pointer(&mut self, position: PhysicalPosition<f64>, rect: DeviceRect) {
        let nx = ((position.x as f32 - rect.center_x()) / (rect.width * 0.5)).clamp(-1.0, 1.0);
        let ny = ((position.y as f32 - rect.center_y()) / (rect.height * 0.5)).clamp(-1.0, 1.0);
        self.target_x = nx;
        self.target_y = ny;
    }

    fn reset_motion(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
        self.target_x = 0.0;
        self.target_y = 0.0;
        self.tilt_deg = self.manifest.motion.tilt_default_deg;
        self.parallax = self.manifest.motion.parallax_default;
    }

    fn window_title(&self) -> String {
        format!(
            "{} · tilt {:.1}° · parallax {:.2} · [[/]] tilt · [-/=] parallax · [0] reset",
            self.title_prefix, self.tilt_deg, self.parallax
        )
    }
}

#[derive(Clone, Copy)]
struct DeviceRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl DeviceRect {
    fn center_x(self) -> f32 {
        self.x + self.width * 0.5
    }

    fn center_y(self) -> f32 {
        self.y + self.height * 0.5
    }

    fn contains(self, pos: PhysicalPosition<f64>) -> bool {
        let x = pos.x as f32;
        let y = pos.y as f32;
        x >= self.x && x <= self.x + self.width && y >= self.y && y <= self.y + self.height
    }
}

struct GpuRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    sampler: wgpu::Sampler,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_pipeline: wgpu::RenderPipeline,
    color_pipeline: wgpu::RenderPipeline,
    texture_vertex_buffer: wgpu::Buffer,
    texture_vertex_capacity: usize,
    color_vertex_buffer: wgpu::Buffer,
    color_vertex_capacity: usize,
}

impl GpuRenderer {
    async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .context("failed to create preview surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .context("failed to acquire GPU adapter for preview")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sharp-mcu-preview device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                ..Default::default()
            })
            .await
            .context("failed to create GPU device for preview")?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .or_else(|| caps.formats.first().copied())
            .context("surface reported no supported formats")?;
        let present_mode = caps
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::AutoVsync)
            .or_else(|| {
                caps.present_modes
                    .iter()
                    .copied()
                    .find(|mode| *mode == wgpu::PresentMode::Fifo)
            })
            .or_else(|| caps.present_modes.first().copied())
            .unwrap_or(wgpu::PresentMode::AutoVsync);
        let alpha_mode = caps
            .alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Auto);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            desired_maximum_frame_latency: 2,
            present_mode,
            alpha_mode,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sharp-mcu-preview sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sharp-mcu-preview texture bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let textured_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sharp-mcu-preview textured shader"),
            source: wgpu::ShaderSource::Wgsl(TEXTURED_SHADER.into()),
        });
        let color_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sharp-mcu-preview color shader"),
            source: wgpu::ShaderSource::Wgsl(COLOR_SHADER.into()),
        });

        let texture_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sharp-mcu-preview texture pipeline layout"),
                bind_group_layouts: &[Some(&texture_bind_group_layout)],
                immediate_size: 0,
            });
        let texture_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sharp-mcu-preview texture pipeline"),
            layout: Some(&texture_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &textured_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[TexturedVertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &textured_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let color_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sharp-mcu-preview color pipeline layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let color_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sharp-mcu-preview color pipeline"),
            layout: Some(&color_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &color_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[ColorVertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &color_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let texture_vertex_capacity = 1024usize;
        let color_vertex_capacity = 1024usize;
        let texture_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sharp-mcu-preview textured vertices"),
            size: texture_vertex_capacity as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let color_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sharp-mcu-preview color vertices"),
            size: color_vertex_capacity as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            sampler,
            texture_bind_group_layout,
            texture_pipeline,
            color_pipeline,
            texture_vertex_buffer,
            texture_vertex_capacity,
            color_vertex_buffer,
            color_vertex_capacity,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) -> Result<()> {
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        Ok(())
    }

    fn upload_layer_texture(&self, rgba: &[u8], width: u32, height: u32) -> Result<LayerTexture> {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sharp-mcu-preview layer texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sharp-mcu-preview layer bind group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        Ok(LayerTexture {
            _texture: texture,
            _view: view,
            bind_group,
        })
    }

    fn render(&mut self, state: &PreviewState) -> Result<()> {
        if self.config.width == 0 || self.config.height == 0 {
            return Ok(());
        }

        let stage = stage_rect(
            self.config.width as f32,
            self.config.height as f32,
            state.manifest.package.width as f32,
            state.manifest.package.height as f32,
        );
        let center_x = stage.center_x();
        let center_y = stage.center_y();
        let rotation_rad = stage_rotation_deg(
            state.x,
            state.y,
            state.tilt_deg,
            &state.manifest.motion.tuning,
        )
        .to_radians();
        let cos = rotation_rad.cos();
        let sin = rotation_rad.sin();

        let mut textured_vertices = Vec::with_capacity(state.layers.len() * 6);
        let mut textured_draws = Vec::with_capacity(state.layers.len());
        for (index, layer) in state.layers.iter().enumerate() {
            let Some(_texture) = layer.texture.as_ref() else {
                continue;
            };
            let transform = compute_layer_transform(
                state.x,
                state.y,
                stage.width,
                stage.height,
                state.parallax,
                &layer.meta.motion,
                &state.manifest.motion.tuning,
            );
            let displayed_w = stage.width * transform.scale;
            let displayed_h = stage.height * transform.scale;
            let cx = center_x + transform.tx_px;
            let cy = center_y + transform.ty_px;
            let start = textured_vertices.len() as u32;
            push_textured_quad(
                &mut textured_vertices,
                self.config.width as f32,
                self.config.height as f32,
                cx,
                cy,
                displayed_w,
                displayed_h,
                cos,
                sin,
            );
            textured_draws.push(TexturedDraw {
                layer_index: index,
                vertex_start: start,
                vertex_count: 6,
            });
        }

        let mut overlay_vertices = Vec::with_capacity(96);
        push_overlay_geometry(
            &mut overlay_vertices,
            self.config.width as f32,
            self.config.height as f32,
            stage,
            device_pad_rect(self.config.width, self.config.height),
            state.x,
            state.y,
        );

        self.ensure_texture_vertex_capacity(
            textured_vertices.len() * std::mem::size_of::<TexturedVertex>(),
        );
        self.ensure_color_vertex_capacity(
            overlay_vertices.len() * std::mem::size_of::<ColorVertex>(),
        );
        if !textured_vertices.is_empty() {
            self.queue.write_buffer(
                &self.texture_vertex_buffer,
                0,
                bytemuck::cast_slice(&textured_vertices),
            );
        }
        if !overlay_vertices.is_empty() {
            self.queue.write_buffer(
                &self.color_vertex_buffer,
                0,
                bytemuck::cast_slice(&overlay_vertices),
            );
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return Ok(()),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sharp-mcu-preview encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sharp-mcu-preview pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            if !textured_draws.is_empty() {
                pass.set_pipeline(&self.texture_pipeline);
                pass.set_vertex_buffer(0, self.texture_vertex_buffer.slice(..));
                for draw in &textured_draws {
                    let texture = state.layers[draw.layer_index]
                        .texture
                        .as_ref()
                        .expect("draw recorded without texture");
                    pass.set_bind_group(0, &texture.bind_group, &[]);
                    pass.draw(
                        draw.vertex_start..draw.vertex_start + draw.vertex_count,
                        0..1,
                    );
                }
            }

            if !overlay_vertices.is_empty() {
                pass.set_pipeline(&self.color_pipeline);
                pass.set_vertex_buffer(0, self.color_vertex_buffer.slice(..));
                pass.draw(0..overlay_vertices.len() as u32, 0..1);
            }
        }

        self.queue.submit([encoder.finish()]);
        frame.present();
        Ok(())
    }

    fn ensure_texture_vertex_capacity(&mut self, needed_bytes: usize) {
        if needed_bytes <= self.texture_vertex_capacity {
            return;
        }
        self.texture_vertex_capacity = needed_bytes.next_power_of_two().max(1024);
        self.texture_vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sharp-mcu-preview textured vertices"),
            size: self.texture_vertex_capacity as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    fn ensure_color_vertex_capacity(&mut self, needed_bytes: usize) {
        if needed_bytes <= self.color_vertex_capacity {
            return;
        }
        self.color_vertex_capacity = needed_bytes.next_power_of_two().max(1024);
        self.color_vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sharp-mcu-preview color vertices"),
            size: self.color_vertex_capacity as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }
}

struct LayerTexture {
    _texture: wgpu::Texture,
    _view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

struct TexturedDraw {
    layer_index: usize,
    vertex_start: u32,
    vertex_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TexturedVertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

impl TexturedVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColorVertex {
    pos: [f32; 2],
    color: [f32; 4],
}

impl ColorVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

enum LayerLoadMessage {
    Loaded {
        index: usize,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    },
    Failed {
        index: usize,
        path: PathBuf,
        error: String,
    },
}

fn spawn_layer_loader(paths: Vec<PathBuf>) -> Receiver<LayerLoadMessage> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for (index, path) in paths.into_iter().enumerate() {
            let message = match load_rgba_image(&path) {
                Ok((width, height, rgba)) => LayerLoadMessage::Loaded {
                    index,
                    width,
                    height,
                    rgba,
                },
                Err(error) => LayerLoadMessage::Failed {
                    index,
                    path,
                    error: error.to_string(),
                },
            };
            if tx.send(message).is_err() {
                break;
            }
        }
    });
    rx
}

fn load_rgba_image(path: &PathBuf) -> Result<(u32, u32, Vec<u8>)> {
    let image = ImageReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode {}", path.display()))?
        .into_rgba8();
    let (width, height) = image.dimensions();
    Ok((width, height, image.into_raw()))
}

fn integrate_loaded_layers(
    rx: &Receiver<LayerLoadMessage>,
    state: &mut PreviewState,
    renderer: &GpuRenderer,
) -> Result<bool> {
    let mut changed = false;
    let mut loaded_this_tick = 0usize;
    let max_per_tick = state.layers.len().max(1).min(2);

    while loaded_this_tick < max_per_tick {
        match rx.try_recv() {
            Ok(LayerLoadMessage::Loaded {
                index,
                width,
                height,
                rgba,
            }) => {
                if let Some(layer) = state.layers.get_mut(index) {
                    layer.texture = Some(renderer.upload_layer_texture(&rgba, width, height)?);
                    changed = true;
                }
                loaded_this_tick += 1;
            }
            Ok(LayerLoadMessage::Failed { index, path, error }) => {
                anyhow::bail!(
                    "failed to load layer #{index} from {}: {error}",
                    path.display()
                );
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => break,
        }
    }

    Ok(changed)
}

fn stage_rect(window_w: f32, window_h: f32, image_w: f32, image_h: f32) -> DeviceRect {
    let margin = 36.0;
    let reserved_right = 240.0;
    let reserved_bottom = 42.0;
    let avail_w = (window_w - reserved_right - margin * 2.0).max(120.0);
    let avail_h = (window_h - reserved_bottom - margin * 2.0).max(120.0);
    let scale = (avail_w / image_w).min(avail_h / image_h);
    let width = image_w * scale;
    let height = image_h * scale;
    DeviceRect {
        x: margin + (avail_w - width) * 0.5,
        y: margin + (avail_h - height) * 0.5,
        width,
        height,
    }
}

fn device_pad_rect(window_w: u32, window_h: u32) -> DeviceRect {
    DeviceRect {
        x: window_w as f32 - 210.0,
        y: window_h as f32 - 240.0,
        width: 172.0,
        height: 196.0,
    }
}

fn push_textured_quad(
    out: &mut Vec<TexturedVertex>,
    viewport_w: f32,
    viewport_h: f32,
    center_x: f32,
    center_y: f32,
    width: f32,
    height: f32,
    cos: f32,
    sin: f32,
) {
    let half_w = width * 0.5;
    let half_h = height * 0.5;
    let corners = [
        rotate_corner(-half_w, -half_h, cos, sin),
        rotate_corner(half_w, -half_h, cos, sin),
        rotate_corner(half_w, half_h, cos, sin),
        rotate_corner(-half_w, half_h, cos, sin),
    ];
    let positions = corners.map(|(x, y)| {
        let px = center_x + x;
        let py = center_y + y;
        [px / viewport_w * 2.0 - 1.0, 1.0 - py / viewport_h * 2.0]
    });

    out.extend_from_slice(&[
        TexturedVertex {
            pos: positions[0],
            uv: [0.0, 0.0],
        },
        TexturedVertex {
            pos: positions[1],
            uv: [1.0, 0.0],
        },
        TexturedVertex {
            pos: positions[2],
            uv: [1.0, 1.0],
        },
        TexturedVertex {
            pos: positions[0],
            uv: [0.0, 0.0],
        },
        TexturedVertex {
            pos: positions[2],
            uv: [1.0, 1.0],
        },
        TexturedVertex {
            pos: positions[3],
            uv: [0.0, 1.0],
        },
    ]);
}

fn rotate_corner(x: f32, y: f32, cos: f32, sin: f32) -> (f32, f32) {
    (x * cos - y * sin, x * sin + y * cos)
}

fn push_overlay_geometry(
    out: &mut Vec<ColorVertex>,
    viewport_w: f32,
    viewport_h: f32,
    stage: DeviceRect,
    pad: DeviceRect,
    nx: f32,
    ny: f32,
) {
    let white_stroke = [1.0, 1.0, 1.0, 0.18];
    let pad_fill = [1.0, 1.0, 1.0, 0.08];
    let pad_screen = [1.0, 1.0, 1.0, 0.04];
    let accent = [0.35, 0.56, 0.93, 0.82];

    push_rect_outline(out, viewport_w, viewport_h, stage, 2.0, white_stroke);

    push_solid_rect(out, viewport_w, viewport_h, pad, pad_fill);
    push_rect_outline(out, viewport_w, viewport_h, pad, 2.0, [1.0, 1.0, 1.0, 0.28]);
    push_solid_rect(
        out,
        viewport_w,
        viewport_h,
        DeviceRect {
            x: pad.x + 18.0,
            y: pad.y + 22.0,
            width: pad.width - 36.0,
            height: pad.height - 48.0,
        },
        pad_screen,
    );

    let cross = 22.0;
    push_solid_rect(
        out,
        viewport_w,
        viewport_h,
        DeviceRect {
            x: pad.center_x() - cross,
            y: pad.center_y() - 0.5,
            width: cross * 2.0,
            height: 1.0,
        },
        [1.0, 1.0, 1.0, 0.28],
    );
    push_solid_rect(
        out,
        viewport_w,
        viewport_h,
        DeviceRect {
            x: pad.center_x() - 0.5,
            y: pad.center_y() - cross,
            width: 1.0,
            height: cross * 2.0,
        },
        [1.0, 1.0, 1.0, 0.28],
    );

    let cursor_x = pad.center_x() + nx * pad.width * 0.28;
    let cursor_y = pad.center_y() + ny * pad.height * 0.28;
    push_solid_rect(
        out,
        viewport_w,
        viewport_h,
        DeviceRect {
            x: cursor_x - 10.0,
            y: cursor_y - 10.0,
            width: 20.0,
            height: 20.0,
        },
        accent,
    );
}

fn push_rect_outline(
    out: &mut Vec<ColorVertex>,
    viewport_w: f32,
    viewport_h: f32,
    rect: DeviceRect,
    thickness: f32,
    color: [f32; 4],
) {
    push_solid_rect(
        out,
        viewport_w,
        viewport_h,
        DeviceRect {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: thickness,
        },
        color,
    );
    push_solid_rect(
        out,
        viewport_w,
        viewport_h,
        DeviceRect {
            x: rect.x,
            y: rect.y + rect.height - thickness,
            width: rect.width,
            height: thickness,
        },
        color,
    );
    push_solid_rect(
        out,
        viewport_w,
        viewport_h,
        DeviceRect {
            x: rect.x,
            y: rect.y,
            width: thickness,
            height: rect.height,
        },
        color,
    );
    push_solid_rect(
        out,
        viewport_w,
        viewport_h,
        DeviceRect {
            x: rect.x + rect.width - thickness,
            y: rect.y,
            width: thickness,
            height: rect.height,
        },
        color,
    );
}

fn push_solid_rect(
    out: &mut Vec<ColorVertex>,
    viewport_w: f32,
    viewport_h: f32,
    rect: DeviceRect,
    color: [f32; 4],
) {
    let x0 = rect.x / viewport_w * 2.0 - 1.0;
    let x1 = (rect.x + rect.width) / viewport_w * 2.0 - 1.0;
    let y0 = 1.0 - rect.y / viewport_h * 2.0;
    let y1 = 1.0 - (rect.y + rect.height) / viewport_h * 2.0;
    out.extend_from_slice(&[
        ColorVertex {
            pos: [x0, y0],
            color,
        },
        ColorVertex {
            pos: [x1, y0],
            color,
        },
        ColorVertex {
            pos: [x1, y1],
            color,
        },
        ColorVertex {
            pos: [x0, y0],
            color,
        },
        ColorVertex {
            pos: [x1, y1],
            color,
        },
        ColorVertex {
            pos: [x0, y1],
            color,
        },
    ]);
}
