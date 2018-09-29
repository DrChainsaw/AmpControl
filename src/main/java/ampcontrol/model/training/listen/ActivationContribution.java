package ampcontrol.model.training.listen;

import ampcontrol.model.training.model.vertex.EpsilonSpyVertexImpl;
import lombok.Getter;
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

    @Getter
    private class Contribution {
        private final WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT) // Because the size is not gonna change?
                .policyLearning(LearningPolicy.OVER_TIME) // To make cyclesBeforeInitialization take effect?
                .cyclesBeforeInitialization(2) // Because after two arrays the size should be fully known?
                //.policyMirroring(MirroringPolicy.HOST_ONLY)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED) // To make it cyclic?
                .policySpill(SpillPolicy.REALLOCATE)  // Does not matter?
                .initialSize(0) // Any benefit over time with > 0?
                .build();

        private INDArray act;
        private final String wsNameIter = "ContributionWs" + this.toString().split("@")[1];


        private void setAct(INDArray act) {
            try (MemoryWorkspace wss = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, this.wsNameIter)) {
                this.act = act.migrate(false);
            }
        }

        private void setEps(INDArray eps) {
            try (MemoryWorkspace wss = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, this.wsNameIter)) {
                final INDArray tmpEps = eps.migrate(false);
                int[] meanDims = IntStream.range(0, getAct().rank()).filter(dim -> dim != 1).toArray();
                listener.accept(tmpEps.muli(act).amean(meanDims));
            }
        }

        protected void destroy() {
            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceConfig, this.wsNameIter).destroyWorkspace(true);
        }
    }

    public ActivationContribution(String layerName, Consumer<INDArray> listener) {
        this.layerName = layerName;
        this.listener = listener;
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        if (lastContribution == null) {
            initEpsilonListener(model);
        }
        lastContribution.setAct(activations.get(layerName));
    }

    @Override
    public void onEpochEnd(Model model) {
        Nd4j.getExecutioner().commit();
        lastContribution.destroy();
        lastContribution = null;
    }

    private void initEpsilonListener(Model model) {
        if (model instanceof ComputationGraph) {
            ComputationGraph cg = ((ComputationGraph) model);
            if (cg.getVertex(layerName).getOutputVertices().length != 1) {
                throw new UnsupportedOperationException("More than one output for " + layerName + "!");
            }

            lastContribution = new Contribution();
            Stream.of(cg.getVertex(layerName).getOutputVertices())
                    .map(VertexIndices::getVertexIndex)
                    .map(outputVertexInd -> cg.getVertices()[outputVertexInd])
                    .filter(graphVertex -> graphVertex instanceof EpsilonSpyVertexImpl)
                    .map(graphVertex -> (EpsilonSpyVertexImpl) graphVertex)
                    .findAny()
                    .orElseThrow(() -> new UnsupportedOperationException("Must have " + EpsilonSpyVertexImpl.class +
                            " as output to " + layerName + "!"))
                    .setListener(lastContribution::setEps);
        } else { // I.e not a ComputationGraph instance
            throw new IllegalArgumentException("Can only work on ComputationGraph instances! Got " + model);
        }
    }
}
