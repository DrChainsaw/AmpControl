package ampcontrol.model.training.listen;

import ampcontrol.model.training.model.vertex.EpsilonSpyVertexImpl;
import lombok.Getter;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
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
    final String wsName = "ContributionWs" + this.toString().split("@")[1];

    @Getter
    private abstract class Contribution {
        private INDArray act;
        private INDArray eps;


        private final WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyMirroring(MirroringPolicy.HOST_ONLY)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                .policySpill(SpillPolicy.REALLOCATE)
                .initialSize(0)
                //.overallocationLimit(20)
                .build();

        abstract INDArray getContrib();

        abstract Contribution calc();

        private void setAct(INDArray act) {
            final MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceConfig, wsName);
            try (MemoryWorkspace wss = ws.notifyScopeEntered()) {
                this.act = act.migrate(false);
            }
        }

        private void setEps(INDArray eps) {
            final MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceConfig, wsName);
            try (MemoryWorkspace wss = ws.notifyScopeEntered()) {
                this.eps = eps.migrate(false);
            }
        }

        protected void destroy() {
           Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceConfig, wsName).destroyWorkspace();
        }
    }

    private class InitialContribution extends Contribution {

        @Override
        public INDArray getContrib() {
            return null;
        }

        public Contribution calc() {
            int[] meanDims = IntStream.range(0, getAct().rank()).filter(dim -> dim != 1).toArray();

            final INDArray contribTemplate = Nd4j.zeros(1, getAct().size(1));

            return new SumContribution(getAct(), getEps(), contribTemplate, meanDims);
        }
    }

    private class SumContribution extends Contribution {

        private final INDArray contribution;
        private final int[] meanDims;

        private SumContribution(INDArray act, INDArray eps, INDArray contribution, int[] meanDims) {
            if (act == null) {
                throw new IllegalStateException("No activation!");
            }
            if (eps == null) {
                throw new IllegalStateException("No epsilon!");
            }
            this.contribution = contribution.addi(eps.muli(act).amean(meanDims));
            this.meanDims = meanDims;
        }

        @Override
        public INDArray getContrib() {
            return contribution;
        }

        @Override
        public Contribution calc() {
            return new SumContribution(getAct(), getEps(), contribution, meanDims);
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
    public void onGradientCalculation(Model model) {
        try {
            lastContribution = lastContribution.calc();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to calculate contribution for layer " + layerName, e);
        }
    }

    @Override
    public void onEpochEnd(Model model) {
        listener.accept(lastContribution.getContrib());
        lastContribution.destroy();
        lastContribution = new InitialContribution();
    }

    private void initEpsilonListener(Model model) {
        if (model instanceof ComputationGraph) {
            ComputationGraph cg = ((ComputationGraph) model);
            if (cg.getVertex(layerName).getOutputVertices().length != 1) {
                throw new UnsupportedOperationException("More than one output for " + layerName + "!");
            }

            lastContribution = new InitialContribution();
            Stream.of(cg.getVertex(layerName).getOutputVertices())
                    .map(VertexIndices::getVertexIndex)
                    .map(outputVertexInd -> cg.getVertices()[outputVertexInd])
                    .filter(graphVertex -> graphVertex instanceof EpsilonSpyVertexImpl)
                    .map(graphVertex -> (EpsilonSpyVertexImpl) graphVertex)
                    .findAny()
                    .orElseThrow(() -> new UnsupportedOperationException("Must have " + EpsilonSpyVertexImpl.class +
                            " as output to " + layerName + "!"))
                    .addListener(epsilon -> lastContribution.setEps(epsilon));
        } else { // I.e not a ComputationGraph instance
            throw new IllegalArgumentException("Can only work on ComputationGraph instances! Got " + model);
        }
    }
}
