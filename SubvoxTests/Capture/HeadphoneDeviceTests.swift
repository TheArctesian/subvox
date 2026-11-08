import XCTest
@testable import Subvox

final class HeadphoneDeviceTests: XCTestCase {
    func testConnectedAirPodsGiveProductFirmwareAndModelButNeverTheName() throws {
        let json = """
        {"SPBluetoothDataType":[{"device_connected":[
          {"Stephen's Keyboard":{"device_minorType":"Keyboard","device_vendorID":"0x004C","device_productID":"0x029C"}},
          {"Stephen's AirPods Pro":{"device_minorType":"Headphones","device_vendorID":"0x004C","device_productID":"0x2027",
                                     "device_firmwareVersion":"9\\u200bA348","device_address":"AA:BB:CC:DD:EE:FF"}}],
          "device_not_connected":[{"Old":{"device_minorType":"Headphones","device_vendorID":"0x004C","device_productID":"0x2014"}}]}]}
        """
        let info = try XCTUnwrap(MotionDeviceInfo.parse(systemProfilerJSON: Data(json.utf8)))
        XCTAssertEqual(info, MotionDeviceInfo(vendorID: "0x004C", productID: "0x2027", model: "AirPods Pro 3", firmware: "9A348"))
        let encoded = String(decoding: try JSONEncoder().encode(info), as: UTF8.self)
        XCTAssertFalse(encoded.contains("Stephen") || encoded.contains("AA:BB"), "No user-set name or Bluetooth address")
    }

    func testUnknownOrMissingHeadphones() {
        let unknown = #"{"SPBluetoothDataType":[{"device_connected":[{"x":{"device_minorType":"Headphones","device_vendorID":"0x004c","device_productID":"0x2999"}}]}]}"#
        XCTAssertEqual(MotionDeviceInfo.parse(systemProfilerJSON: Data(unknown.utf8))?.model, "Apple headphones 0x2999")
        let none = #"{"SPBluetoothDataType":[{"device_connected":[{"x":{"device_minorType":"Headset","device_vendorID":"0x054C","device_productID":"0x0D58"}}]}]}"#
        XCTAssertNil(MotionDeviceInfo.parse(systemProfilerJSON: Data(none.utf8)), "Non-Apple headsets have no headphone motion")
    }
}
