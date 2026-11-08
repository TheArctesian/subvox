import CoreMotion
import Observation
import os

/// Streams `CMHeadphoneMotionManager` updates from connected AirPods.
@MainActor @Observable
final class HeadphoneMotionService {
    enum Status: Equatable {
        case unsupported
        case denied
        case waiting
        case streaming
        case error(String)

        var label: String {
            switch self {
            case .unsupported: String(localized: "Headphone motion unsupported")
            case .denied: String(localized: "Motion access denied")
            case .waiting: String(localized: "Waiting for AirPods")
            case .streaming: String(localized: "Streaming")
            case .error(let message): message
            }
        }
    }

    static let historySeconds: TimeInterval = 10
    /// Connected but silent for this long: restart the stream (AirPods can stop sending after a route change).
    static let stallSeconds: TimeInterval = 2

    /// What the user should see, including the "connected but no motion" case the plain status hides.
    var statusText: String {
        switch status {
        case .waiting where isConnected:
            CMHeadphoneMotionManager.authorizationStatus() == .notDetermined
                ? String(localized: "AirPods connected · allow Motion access") : String(localized: "AirPods connected · no motion yet")
        case .waiting: String(localized: "Waiting for AirPods")
        default: status.label
        }
    }
    private(set) var restarts = 0
    @ObservationIgnored private let log = Logger(subsystem: "org.subvox", category: "headphone-motion")
    @ObservationIgnored private var watchdog: Task<Void, Never>?
    @ObservationIgnored private var lastSampleUptime: TimeInterval = 0

    private(set) var status: Status = .waiting
    private(set) var isConnected = false
    private(set) var latest: MotionSample?
    private(set) var history: [MotionSample] = []
    private(set) var sampleRate: Double = 0
    private(set) var sampleCount = 0

    var onSample: ((MotionSample) -> Void)?

    @ObservationIgnored private let manager = CMHeadphoneMotionManager()
    @ObservationIgnored private let proxy = DelegateProxy()
    @ObservationIgnored private var rateWindow: [TimeInterval] = []

    func start() {
        guard manager.isDeviceMotionAvailable else {
            status = .unsupported
            return
        }
        switch CMHeadphoneMotionManager.authorizationStatus() {
        case .denied, .restricted:
            status = .denied
            return
        default:
            break
        }

        proxy.onChange = { [weak self] connected in
            Task { @MainActor in self?.connectionChanged(connected) }
        }
        manager.delegate = proxy
        manager.startConnectionStatusUpdates()
        log.info("start: authorization \(CMHeadphoneMotionManager.authorizationStatus().rawValue, privacy: .public)")
        startUpdates()
        watchdog?.cancel()
        watchdog = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                self?.checkStall()
            }
        }
    }

    private func startUpdates() {
        manager.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            MainActor.assumeIsolated {
                self?.handle(motion, error)
            }
        }
    }

    private func restartUpdates(reason: String) {
        restarts += 1
        log.notice("restart motion updates (\(reason, privacy: .public)), restart #\(self.restarts)")
        manager.stopDeviceMotionUpdates()
        startUpdates()
        lastSampleUptime = ProcessInfo.processInfo.systemUptime
    }

    /// Connected but no samples for a while: CoreMotion sometimes keeps the connection yet stops
    /// delivering after the AirPods switch route or wake. Restarting the stream recovers it.
    private func checkStall() {
        let now = ProcessInfo.processInfo.systemUptime
        guard isConnected, now - lastSampleUptime > Self.stallSeconds else { return }
        if status == .streaming { log.notice("motion stalled for \(now - self.lastSampleUptime, privacy: .public)s while connected") }
        status = .waiting
        sampleRate = 0
        rateWindow.removeAll()
        restartUpdates(reason: "stall")
    }

    func stop() {
        watchdog?.cancel()
        manager.stopDeviceMotionUpdates()
        manager.stopConnectionStatusUpdates()
    }

    private func connectionChanged(_ connected: Bool) {
        log.notice("AirPods \(connected ? "connected" : "disconnected", privacy: .public)")
        isConnected = connected
        if connected {
            restartUpdates(reason: "reconnected")
        } else {
            status = .waiting
            sampleRate = 0
            rateWindow.removeAll()
        }
    }

    private func handle(_ motion: CMDeviceMotion?, _ error: Error?) {
        if let error {
            let ns = error as NSError
            log.error("motion error \(ns.domain, privacy: .public) \(ns.code): \(ns.localizedDescription, privacy: .public)")
            if ns.domain == CMErrorDomain, ns.code == Int(CMErrorMotionActivityNotAuthorized.rawValue) {
                status = .denied
            } else {
                status = .error(error.localizedDescription)
            }
            return
        }
        guard let motion else { return }

        let sample = MotionSample(motion)
        lastSampleUptime = ProcessInfo.processInfo.systemUptime
        if status != .streaming { log.notice("motion streaming (sensor \(motion.sensorLocation.rawValue, privacy: .public))") }
        latest = sample
        status = .streaming
        isConnected = true
        sampleCount += 1

        history.append(sample)
        let cutoff = sample.t - Self.historySeconds
        if let firstKept = history.firstIndex(where: { $0.t >= cutoff }), firstKept > 0 {
            history.removeFirst(firstKept)
        }

        rateWindow.append(sample.t)
        rateWindow.removeAll { $0 < sample.t - 1 }
        if let first = rateWindow.first, sample.t > first {
            sampleRate = Double(rateWindow.count - 1) / (sample.t - first)
        }

        onSample?(sample)
    }
}

private final class DelegateProxy: NSObject, CMHeadphoneMotionManagerDelegate {
    var onChange: ((Bool) -> Void)?

    func headphoneMotionManagerDidConnect(_ manager: CMHeadphoneMotionManager) {
        onChange?(true)
    }

    func headphoneMotionManagerDidDisconnect(_ manager: CMHeadphoneMotionManager) {
        onChange?(false)
    }
}
