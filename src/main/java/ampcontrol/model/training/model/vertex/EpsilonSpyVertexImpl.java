package ampcontrol.model.training.model.vertex;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * {@link BaseGraphVertex} which spies on Epsilons since it is not possible to obtain them through a listener. Note:
 * Since vertexes must be deserializable the listeners must be added once the graph is built, most likely forcing
 * a type cast to do so.
 *
 * @author Christian Skärby
 */
public class EpsilonSpyVertexImpl extends BaseGraphVertex {

    private final List<Consumer<INDArray>> listeners = new ArrayList<>();

    protected EpsilonSpyVertexImpl(ComputationGraph graph, String name, int vertexIndex) {
        this(graph, name, vertexIndex, null, null);
    }

    protected EpsilonSpyVertexImpl(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
    }

    public void addListener(Consumer<INDArray> listener) {
        listeners.add(listener);
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\")";
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) { return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, inputs[0]);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        listeners.forEach(listener -> listener.accept(getEpsilon()));
        return new Pair<>(null, new INDArray[]{workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilon)});
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new IllegalArgumentException(
                    "Vertex does not have gradients; gradients view array cannot be set here " + this.toString());
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        //No op
        if (maskArrays == null || maskArrays.length == 0) {
            return null;
        }

        return new Pair<>(maskArrays[0], currentMaskState);
    }
}
