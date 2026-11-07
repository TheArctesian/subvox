import Foundation

enum Alignment {
    static func spread(weights: [Double], over range: ClosedRange<Double>) -> [ClosedRange<Double>] {
        let total = weights.reduce(0, +)
        guard total > 0 else { return weights.map { _ in range } }
        var cursor = range.lowerBound
        let length = range.upperBound - range.lowerBound
        return weights.map { weight in
            let end = cursor + length * weight / total
            defer { cursor = end }
            return cursor...end
        }
    }

    /// Index pairs of a longest common subsequence of two word lists.
    static func commonSubsequence(_ a: [String], _ b: [String]) -> [(Int, Int)] {
        guard !a.isEmpty, !b.isEmpty else { return [] }
        var table = [[Int]](repeating: [Int](repeating: 0, count: b.count + 1), count: a.count + 1)
        for i in stride(from: a.count - 1, through: 0, by: -1) {
            for j in stride(from: b.count - 1, through: 0, by: -1) {
                table[i][j] = a[i] == b[j] ? table[i + 1][j + 1] + 1 : max(table[i + 1][j], table[i][j + 1])
            }
        }
        var pairs: [(Int, Int)] = []
        var i = 0, j = 0
        while i < a.count, j < b.count {
            if a[i] == b[j] {
                pairs.append((i, j))
                i += 1
                j += 1
            } else if table[i + 1][j] >= table[i][j + 1] {
                i += 1
            } else {
                j += 1
            }
        }
        return pairs
    }
}
