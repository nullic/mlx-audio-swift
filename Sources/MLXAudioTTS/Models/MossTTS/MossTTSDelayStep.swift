@preconcurrency import MLX
import MLXNN

/// The delay loop's per-step work without what it does not need: the 32
/// audio embeddings and heads run as one gather and one batched product, and
/// inside audio the text channel may only pick one of two tokens, so only
/// those two rows of the text head are multiplied. The per-codebook modules
/// are then pointed at slices of the joined arrays, so nothing is held twice.
final class MossTTSDelayStep {
    final class Memo {
        var value: MossTTSDelayStep?
    }

    private let textEmbedding: Embedding
    private let audioEmbeddings: [Embedding]
    private let audioEmbeddingTable: MLXArray?
    private let text: Linear
    private let audioHeads: [Linear]
    private let insideAudioIDs: MLXArray
    private let insideAudioWeight: MLXArray?
    private let audioWeight: MLXArray?

    init(textEmbedding: Embedding, audioEmbeddings: [Embedding], text: Linear, audio: [Linear], insideAudio: [Int]) {
        self.textEmbedding = textEmbedding
        self.audioEmbeddings = audioEmbeddings
        self.audioEmbeddingTable = audioEmbeddings.contains { $0 is QuantizedEmbedding }
            ? nil
            : MLX.concatenated(audioEmbeddings.map(\.weight), axis: 0)
        self.text = text
        self.audioHeads = audio
        self.insideAudioIDs = MLXArray(insideAudio.map(Int32.init))
        self.insideAudioWeight = text is QuantizedLinear ? nil : text.weight[insideAudioIDs]
        self.audioWeight = audio.contains { $0 is QuantizedLinear } ? nil : MLX.stacked(audio.map(\.weight))
        eval([audioEmbeddingTable, insideAudioWeight, audioWeight].compactMap(\.self))
        if let audioEmbeddingTable {
            let rows = audioEmbeddingTable.dim(0) / max(audioEmbeddings.count, 1)
            for (index, embedding) in audioEmbeddings.enumerated() {
                embedding.update(parameters: ModuleParameters.unflattened(["weight": audioEmbeddingTable[(index * rows) ..< ((index + 1) * rows)]]))
            }
        }
        if let audioWeight {
            for (index, head) in audio.enumerated() {
                head.update(parameters: ModuleParameters.unflattened(["weight": audioWeight[index]]))
            }
        }
    }

    /// `[1, 1, hidden]` for one step's `[1, 1, 1 + codebooks]` tokens. The rows
    /// come in one gather but are added one at a time, in order, as
    /// `buildInputsEmbeds` does: summed in one reduction the bf16 total rounds
    /// differently, and that alone makes the model miss its end and babble.
    func embed(_ tokens: [Int32]) -> MLXArray {
        let textPart = textEmbedding(MLXArray([tokens[0]]).reshaped([1, 1]))
        guard let audioEmbeddingTable else {
            return audioEmbeddings.enumerated().reduce(textPart) { sum, pair in
                sum + pair.element(MLXArray([tokens[pair.offset + 1]]).reshaped([1, 1]))
            }
        }
        let rows = audioEmbeddings.first?.weight.dim(0) ?? 0
        let offsets = tokens.dropFirst().enumerated().map { Int32($0.offset * rows) + $0.element }
        let gathered = audioEmbeddingTable[MLXArray(offsets)]
        return (0 ..< gathered.dim(0)).reduce(textPart) { $0 + gathered[$1].reshaped([1, 1, -1]) }
    }

    /// `[1, 2]`: the text logits of the only two tokens allowed inside audio.
    func insideAudioText(_ hidden: MLXArray) -> MLXArray {
        guard let insideAudioWeight else { return text(hidden)[0..., insideAudioIDs] }
        return matmul(hidden, insideAudioWeight.transposed())
    }

    /// `[codebooks, audioVocab + 1]`, in the order the codebooks are asked for.
    func audio(_ hidden: MLXArray, codebooks: [Int]) -> MLXArray {
        guard let audioWeight else {
            return MLX.concatenated(codebooks.map { audioHeads[$0](hidden) }, axis: 0)
        }
        let logits = matmul(audioWeight, hidden.reshaped([1, -1, 1])).squeezed(axis: -1)
        guard codebooks.count < audioHeads.count else { return logits }
        return logits[MLXArray(codebooks.map(Int32.init))]
    }
}
