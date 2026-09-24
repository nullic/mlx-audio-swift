import Foundation
@preconcurrency import MLX
import MLXAudioCodecs

/// Turns the delay-patterned rows of a MOSS generation into audio while it
/// runs. A frame is finished once its last codebook has been drawn, which
/// `advance` notices after every step; finished frames are decoded in chunks.
/// Segments and the prompt trim follow `decodeGeneratedAudio` exactly: rows
/// that are all padding split segments, and the first segment loses as many
/// frames as the prompt held.
final class MossTTSAudioStreamer {
    private let startIndex: Int
    private let codebooks: Int
    private let padCode: Int32
    private let framesPerChunk: Int
    private let makeDecoder: () throws -> MossAudioDecodeStream
    private let emit: (MLXArray) -> Void

    private var trim: Int
    private var examined = 0
    private var pending: [Int32] = []
    private var pendingFrames = 0
    private var decoder: MossAudioDecodeStream?
    private var inSegment = false
    private var finishedSegments = 0
    private var emitted = 0

    /// The prompt's frames go with the first chunk, so the codec reads them in
    /// the same call; after that each chunk grows by half, up to two seconds —
    /// a decode costs about the same for five frames as for twenty-five, and
    /// half again is what the next chunk can be made in while this one plays.
    private var chunk: Int {
        guard trim == 0 else { return trim + framesPerChunk }
        return min(Int((Double(framesPerChunk) * pow(1.5, Double(min(emitted, 6)))).rounded()), Self.longestChunk)
    }

    private static let longestChunk = 25

    init(
        startIndex: Int,
        trim: Int,
        codebooks: Int,
        padCode: Int32,
        framesPerChunk: Int,
        decoder: @escaping () throws -> MossAudioDecodeStream,
        emit: @escaping (MLXArray) -> Void
    ) {
        self.startIndex = startIndex
        self.trim = trim
        self.codebooks = codebooks
        self.padCode = padCode
        self.framesPerChunk = framesPerChunk
        self.makeDecoder = decoder
        self.emit = emit
    }

    /// `rows` is the whole generation so far, flattened `[rows, 1 + codebooks]`.
    func advance(_ rows: [Int32]) throws {
        let width = codebooks + 1
        let finished = rows.count / width - startIndex - codebooks + 1
        if finished > examined {
            for frame in examined ..< finished {
                var codes = [Int32](repeating: 0, count: codebooks)
                var padding = true
                for codebook in 0 ..< codebooks {
                    let code = rows[(startIndex + frame + codebook) * width + 1 + codebook]
                    codes[codebook] = code
                    if code != padCode { padding = false }
                }
                if padding {
                    if inSegment { try endSegment() }
                    continue
                }
                inSegment = true
                pending += codes
                pendingFrames += 1
            }
            examined = finished
        }
        if pendingFrames >= chunk { try flush() }
    }

    func finish() throws {
        try flush()
    }

    private func endSegment() throws {
        try flush()
        decoder = nil
        inSegment = false
        finishedSegments += 1
    }

    private func flush() throws {
        guard pendingFrames > 0 else { return }
        let decoder = try decoder ?? makeDecoder()
        self.decoder = decoder
        var audio = try decoder.decode(MLXArray(pending, [pendingFrames, codebooks]))
        if finishedSegments == 0, trim > 0 {
            let samplesPerFrame = audio.dim(0) / pendingFrames
            let dropped = min(trim, pendingFrames)
            trim -= dropped
            audio = audio[(dropped * samplesPerFrame)..., 0...]
        }
        pending.removeAll(keepingCapacity: true)
        pendingFrames = 0
        if audio.dim(0) > 0 {
            emitted += 1
            emit(audio)
        }
    }
}
