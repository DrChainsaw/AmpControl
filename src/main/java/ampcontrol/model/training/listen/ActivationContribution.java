package ampcontrol.model.training.listen;

import ampcontrol.model.training.model.vertex.EpsilonSpyVertexImpl;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Calculates activation contribution from individual neurons in a layer according to https://arxiv.org/abs/1611.06440.
 * <br><br>
 * Short summary is that the first order taylor approximation of the optimization problem "which neurons shall I remove
 * to minimize impact on the loss function" boils down to "the ones which minimize abs(gradient * activation)" (assuming
 * parameter independence).
 * <br><br>
 * Note that it is not possible to get the gradient of same shape as activation for convolution layers through a
 * {@link org.deeplearning4j.optimize.api.TrainingListener} in DL4J. As a workaround, this class assumes that the layer
 * after the target layer is an instance of {@link EpsilonSpyVertexImpl} and will register a listener if that is the
 * case, otherwise an exception is thrown.
 *
 * @author Christian Skärby
 */
public class ActivationContribution extends BaseTrainingListener {

    private final String layerName;
    private final Consumer<INDArray> listener;
    private Contribution lastContribution;

    private class Contribution {
        private final WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                //.policyMirroring(MirroringPolicy.HOST_ONLY)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)  // Does not matter?
                .initialSize(0) // Any benefit over time with > 0?
                .build();

        private INDArray act;
        private static final String wsName = "ContributionWs";


        private void setAct(INDArray act) {
            this.act = act;
        }

        private void setEps(INDArray eps) {
            try (MemoryWorkspace wss = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, wsName)) {
                final INDArray tmpEps = eps.migrate(false);
                int[] meanDims = IntStream.range(0, act.rank()).filter(dim -> dim != 1).toArray();
                listener.accept(tmpEps.muli(act).amean(meanDims));
            }
        }
    }

    public ActivationContribution(String layerName, Consumer<INDArray> listener) {
        this.layerName = layerName;
        this.listener = listener;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        super.iterationDone(model, iteration, epoch);
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        if (lastContribution == null) {
            lastContribution = new Contribution();
            setEpsilonListener(model, lastContribution::setEps);
        }
        lastContribution.setAct(activations.get(layerName));

    }

    @Override
    public void onEpochEnd(Model model) {
        lastContribution = null;
    }

    private void setEpsilonListener(Model model, Consumer<INDArray> epsListener) {
        if (model instanceof ComputationGraph) {
            ComputationGraph cg = ((ComputationGraph) model);
            if (cg.getVertex(layerName).getOutputVertices().length != 1) {
                throw new UnsupportedOperationException("More than one output for " + layerName + "!");
            }

            Stream.of(cg.getVertex(layerName).getOutputVertices())
                    .map(VertexIndices::getVertexIndex)
                    .map(outputVertexInd -> cg.getVertices()[outputVertexInd])
                    .filter(graphVertex -> graphVertex instanceof EpsilonSpyVertexImpl)
                    .map(graphVertex -> (EpsilonSpyVertexImpl) graphVertex)
                    .findAny()
                    .orElseThrow(() -> new UnsupportedOperationException("Must have " + EpsilonSpyVertexImpl.class +
                            " as output to " + layerName + "!"))
                    .setListener(epsListener);
        } else { // I.e not a ComputationGraph instance
            throw new IllegalArgumentException("Can only work on ComputationGraph instances! Got " + model);
        }
    }
}
