import Foundation

/// Which headphones produced the motion. AirPods models mount the sensor differently and report at
/// different rates, so this travels with every recording. Only Apple's product and vendor IDs and
/// the firmware version are kept: never the user-assigned device name or Bluetooth address.
struct MotionDeviceInfo: Codable, Equatable, Sendable {
    var vendorID: String
    var productID: String
    var model: String
    var firmware: String?

    /// Best-effort names for Apple product IDs; the raw product ID is the authoritative field.
    static let knownModels: [String: String] = [
        "0x2002": "AirPods (1st generation)", "0x200F": "AirPods (2nd generation)", "0x2013": "AirPods (3rd generation)",
        "0x2019": "AirPods 4", "0x201B": "AirPods 4 (ANC)", "0x200E": "AirPods Pro", "0x2014": "AirPods Pro (2nd generation)",
        "0x2024": "AirPods Pro (2nd generation, USB-C)", "0x2027": "AirPods Pro 3", "0x200A": "AirPods Max",
        "0x201F": "AirPods Max (USB-C)",
    ]

    /// Connected Apple headphones from `system_profiler SPBluetoothDataType -json` output.
    static func parse(systemProfilerJSON data: Data) -> MotionDeviceInfo? {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let controllers = root["SPBluetoothDataType"] as? [[String: Any]] else { return nil }
        for controller in controllers {
            for entry in controller["device_connected"] as? [[String: Any]] ?? [] {
                for case let info as [String: Any] in entry.values {
                    guard let vendor = (info["device_vendorID"] as? String)?.uppercased().replacingOccurrences(of: "0X", with: "0x"),
                          vendor == "0x004C", let product = info["device_productID"] as? String,
                          ["Headphones", "Headset"].contains(info["device_minorType"] as? String ?? "") else { continue }
                    let id = "0x" + product.uppercased().replacingOccurrences(of: "0X", with: "")
                    let firmware = (info["device_firmwareVersion"] as? String)?
                        .unicodeScalars.filter { $0.properties.isAlphabetic || $0.properties.numericType != nil || $0 == "." }
                        .map(String.init).joined()
                    return MotionDeviceInfo(vendorID: vendor, productID: id, model: knownModels[id] ?? "Apple headphones \(id)",
                                            firmware: firmware?.isEmpty == false ? firmware : nil)
                }
            }
        }
        return nil
    }

    /// Runs system_profiler off the main actor. Returns nil when no Apple headphones are connected.
    static func probe() async -> MotionDeviceInfo? {
        await Task.detached(priority: .utility) {
            let process = Process(), pipe = Pipe()
            process.executableURL = URL(fileURLWithPath: "/usr/sbin/system_profiler")
            process.arguments = ["SPBluetoothDataType", "-json"]
            process.standardOutput = pipe
            process.standardError = FileHandle.nullDevice
            guard (try? process.run()) != nil else { return nil }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            return parse(systemProfilerJSON: data)
        }.value
    }
}
