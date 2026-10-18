import Foundation
import Vision
import AppKit

struct OCRItem: Codable {
    let text: String
    let x: Double
    let y: Double
    let w: Double
    let h: Double
}

func ocr(imagePath: String) -> [OCRItem] {
    guard let image = NSImage(contentsOfFile: imagePath),
          let tiffData = image.tiffRepresentation,
          let bitmapImage = NSBitmapImageRep(data: tiffData),
          let cgImage = bitmapImage.cgImage else {
        return []
    }
    
    var results: [OCRItem] = []
    let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let request = VNRecognizeTextRequest { (req, err) in
        guard let observations = req.results as? [VNRecognizedTextObservation] else { return }
        let imgW = Double(cgImage.width)
        let imgH = Double(cgImage.height)
        for obs in observations {
            if let candidate = obs.topCandidates(1).first {
                let box = obs.boundingBox
                let x = box.origin.x * imgW
                let y = (1.0 - box.origin.y - box.size.height) * imgH
                let w = box.size.width * imgW
                let h = box.size.height * imgH
                results.append(OCRItem(text: candidate.string, x: x, y: y, w: w, h: h))
            }
        }
    }
    request.recognitionLevel = .accurate
    try? requestHandler.perform([request])
    return results
}

let args = CommandLine.arguments
if args.count > 1 {
    let path = args[1]
    let items = ocr(imagePath: path)
    if let data = try? JSONEncoder().encode(items),
       let str = String(data: data, encoding: .utf8) {
        print(str)
    }
}
