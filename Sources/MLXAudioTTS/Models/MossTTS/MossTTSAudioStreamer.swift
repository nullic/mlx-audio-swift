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

    /// `generated` is the whole generation so far, `[rows, 1 + codebooks]`.
    func advance(_ generated: MLXArray) throws {
        let finished = generated.dim(0) - startIndex - codebooks + 1
        if finished > examined {
            let rows = generated[(startIndex + examined) ..< (startIndex + finished + codebooks - 1), 1...]
                .asType(.int32)
                .asArray(Int32.self)
            for frame in 0 ..< (finished - examined) {
                var codes = [Int32](repeating: 0, count: codebooks)
                var padding = true
                for codebook in 0 ..< codebooks {
                    let code = rows[(frame + codebook) * codebooks + codebook]
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
        if pendingFrames >= framesPerChunk { try flush() }
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
        if audio.dim(0) > 0 { emit(audio) }
    }
}
