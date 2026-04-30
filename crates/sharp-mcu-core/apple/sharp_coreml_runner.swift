import Foundation
import CoreML
import CoreImage
import AppKit

struct TensorMeta: Codable {
    let name: String
    let file: String
    let shape: [Int]
}

struct OutputManifest: Codable {
    let tensors: [TensorMeta]
}

final class SHARPCoreMLRunner {
    private let model: MLModel
    private let inputHeight: Int
    private let inputWidth: Int

    init(modelPath: URL, compiledCacheDir: URL, inputHeight: Int = 1536, inputWidth: Int = 1536) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let compiledURL = try Self.compileModelIfNeeded(at: modelPath, cacheDir: compiledCacheDir)
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
    }

    private static func compileModelIfNeeded(at modelPath: URL, cacheDir: URL) throws -> URL {
        let fileManager = FileManager.default
        let ext = modelPath.pathExtension.lowercased()
        if ext == "mlmodelc" {
            return modelPath
        }
        guard ext == "mlpackage" || ext == "mlmodel" else {
            throw NSError(
                domain: "SHARPCoreMLRunner",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported CoreML model format: \(ext)"]
            )
        }

        try fileManager.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let compiledPath = cacheDir.appendingPathComponent(modelPath.deletingPathExtension().lastPathComponent + ".mlmodelc")

        if fileManager.fileExists(atPath: compiledPath.path) {
            let sourceAttrs = try fileManager.attributesOfItem(atPath: modelPath.path)
            let compiledAttrs = try fileManager.attributesOfItem(atPath: compiledPath.path)
            if let sourceDate = sourceAttrs[.modificationDate] as? Date,
               let compiledDate = compiledAttrs[.modificationDate] as? Date,
               compiledDate >= sourceDate {
                return compiledPath
            }
            try? fileManager.removeItem(at: compiledPath)
        }

        let temporaryCompiledURL = try MLModel.compileModel(at: modelPath)
        try? fileManager.removeItem(at: compiledPath)
        try fileManager.moveItem(at: temporaryCompiledURL, to: compiledPath)
        return compiledPath
    }

    func preprocessImage(at imagePath: URL) throws -> MLMultiArray {
        guard let nsImage = NSImage(contentsOf: imagePath) else {
            throw NSError(
                domain: "SHARPCoreMLRunner",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to load image from \(imagePath.path)"]
            )
        }
        guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(
                domain: "SHARPCoreMLRunner",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Failed to convert image to CGImage"]
            )
        }

        let ciImage = CIImage(cgImage: cgImage)
        let context = CIContext()
        let scaleX = CGFloat(inputWidth) / ciImage.extent.width
        let scaleY = CGFloat(inputHeight) / ciImage.extent.height
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        guard let resizedCGImage = context.createCGImage(
            scaledImage,
            from: CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight)
        ) else {
            throw NSError(
                domain: "SHARPCoreMLRunner",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Failed to resize image"]
            )
        }

        let imageArray = try MLMultiArray(
            shape: [1, 3, NSNumber(value: inputHeight), NSNumber(value: inputWidth)],
            dataType: .float32
        )

        let width = resizedCGImage.width
        let height = resizedCGImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let bitmapContext = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw NSError(
                domain: "SHARPCoreMLRunner",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create bitmap context"]
            )
        }

        bitmapContext.draw(resizedCGImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        let ptr = imageArray.dataPointer.assumingMemoryBound(to: Float.self)
        let channelStride = inputHeight * inputWidth
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x * bytesPerPixel
                let r = Float(pixelData[pixelIndex]) / 255.0
                let g = Float(pixelData[pixelIndex + 1]) / 255.0
                let b = Float(pixelData[pixelIndex + 2]) / 255.0
                let spatialIndex = y * inputWidth + x
                ptr[0 * channelStride + spatialIndex] = r
                ptr[1 * channelStride + spatialIndex] = g
                ptr[2 * channelStride + spatialIndex] = b
            }
        }
        return imageArray
    }

    func predict(image: MLMultiArray, disparityFactor: Float) throws -> [String: MLMultiArray] {
        let disparityArray = try MLMultiArray(shape: [1], dataType: .float32)
        disparityArray[0] = NSNumber(value: disparityFactor)

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: image),
            "disparity_factor": MLFeatureValue(multiArray: disparityArray),
        ])

        let output = try model.prediction(from: inputFeatures)
        let outputNames = Array(model.modelDescription.outputDescriptionsByName.keys)

        func findOutput(containing keywords: [String]) -> MLMultiArray? {
            for name in outputNames {
                let lower = name.lowercased()
                if keywords.contains(where: { lower.contains($0.lowercased()) }) {
                    return output.featureValue(for: name)?.multiArrayValue
                }
            }
            return nil
        }

        let exactOrMatched: [(String, MLMultiArray?)] = [
            ("mean_vectors_3d_positions", output.featureValue(for: "mean_vectors_3d_positions")?.multiArrayValue ?? findOutput(containing: ["mean", "position", "xyz"])),
            ("singular_values_scales", output.featureValue(for: "singular_values_scales")?.multiArrayValue ?? findOutput(containing: ["singular", "scale"])),
            ("quaternions_rotations", output.featureValue(for: "quaternions_rotations")?.multiArrayValue ?? findOutput(containing: ["quaternion", "rotation", "rot"])),
            ("colors_rgb_linear", output.featureValue(for: "colors_rgb_linear")?.multiArrayValue ?? findOutput(containing: ["color", "rgb"])),
            ("opacities_alpha_channel", output.featureValue(for: "opacities_alpha_channel")?.multiArrayValue ?? findOutput(containing: ["opacity", "alpha"])),
        ]

        if exactOrMatched.allSatisfy({ $0.1 != nil }) {
            var result: [String: MLMultiArray] = [:]
            for (name, value) in exactOrMatched {
                result[name] = value!
            }
            return result
        }

        if outputNames.count >= 5 {
            let sortedNames = outputNames.sorted()
            guard
                let mv = output.featureValue(for: sortedNames[0])?.multiArrayValue,
                let sv = output.featureValue(for: sortedNames[1])?.multiArrayValue,
                let q = output.featureValue(for: sortedNames[2])?.multiArrayValue,
                let c = output.featureValue(for: sortedNames[3])?.multiArrayValue,
                let o = output.featureValue(for: sortedNames[4])?.multiArrayValue
            else {
                throw NSError(
                    domain: "SHARPCoreMLRunner",
                    code: 6,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to extract model outputs"]
                )
            }
            return [
                "mean_vectors_3d_positions": mv,
                "singular_values_scales": sv,
                "quaternions_rotations": q,
                "colors_rgb_linear": c,
                "opacities_alpha_channel": o,
            ]
        }

        throw NSError(
            domain: "SHARPCoreMLRunner",
            code: 7,
            userInfo: [NSLocalizedDescriptionKey: "Could not match CoreML outputs by name"]
        )
    }
}

func writeTensor(_ array: MLMultiArray, name: String, outputDir: URL) throws -> TensorMeta {
    let count = array.count
    var values = [Float](repeating: 0, count: count)

    switch array.dataType {
    case .float32:
        let src = array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            values[i] = src[i]
        }
    case .double:
        let src = array.dataPointer.assumingMemoryBound(to: Double.self)
        for i in 0..<count {
            values[i] = Float(src[i])
        }
    case .float16:
        let src = array.dataPointer.assumingMemoryBound(to: UInt16.self)
        for i in 0..<count {
            values[i] = Float(Float16(bitPattern: src[i]))
        }
    case .int32:
        let src = array.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<count {
            values[i] = Float(src[i])
        }
    case .int8:
        let src = array.dataPointer.assumingMemoryBound(to: Int8.self)
        for i in 0..<count {
            values[i] = Float(src[i])
        }
    @unknown default:
        throw NSError(
            domain: "SHARPCoreMLRunner",
            code: 8,
            userInfo: [NSLocalizedDescriptionKey: "Unsupported MLMultiArray data type"]
        )
    }

    let file = "\(name).bin"
    let fileURL = outputDir.appendingPathComponent(file)
    let data = values.withUnsafeBufferPointer { ptr in
        Data(buffer: ptr)
    }
    try data.write(to: fileURL)
    return TensorMeta(name: name, file: file, shape: array.shape.map { $0.intValue })
}

func main() {
    let args = CommandLine.arguments
    guard args.count == 6 else {
        fputs("usage: sharp_coreml_runner <model_path> <image_path> <output_dir> <disparity_factor> <compiled_cache_dir>\\n", stderr)
        exit(2)
    }

    let modelPath = URL(fileURLWithPath: args[1])
    let imagePath = URL(fileURLWithPath: args[2])
    let outputDir = URL(fileURLWithPath: args[3], isDirectory: true)
    let disparityFactor = Float(args[4]) ?? 1.0
    let compiledCacheDir = URL(fileURLWithPath: args[5], isDirectory: true)

    do {
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let runner = try SHARPCoreMLRunner(modelPath: modelPath, compiledCacheDir: compiledCacheDir)
        let image = try runner.preprocessImage(at: imagePath)
        let outputs = try runner.predict(image: image, disparityFactor: disparityFactor)

        let orderedNames = [
            "mean_vectors_3d_positions",
            "singular_values_scales",
            "quaternions_rotations",
            "colors_rgb_linear",
            "opacities_alpha_channel",
        ]
        let tensors = try orderedNames.map { name in
            guard let array = outputs[name] else {
                throw NSError(
                    domain: "SHARPCoreMLRunner",
                    code: 9,
                    userInfo: [NSLocalizedDescriptionKey: "Missing required output tensor \(name)"]
                )
            }
            return try writeTensor(array, name: name, outputDir: outputDir)
        }

        let manifest = OutputManifest(tensors: tensors)
        let manifestURL = outputDir.appendingPathComponent("outputs.json")
        let manifestData = try JSONEncoder().encode(manifest)
        try manifestData.write(to: manifestURL)
    } catch {
        fputs("error: \(error.localizedDescription)\\n", stderr)
        exit(1)
    }
}

main()
