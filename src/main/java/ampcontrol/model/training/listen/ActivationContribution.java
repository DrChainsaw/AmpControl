package ampcontrol.model.training.listen;

import ampcontrol.model.training.model.vertex.EpsilonSpyVertexImpl;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
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


    @Getter @Setter
    private abstract static class Contribution {
        private INDArray act;
        private INDArray eps;

        abstract INDArray getContrib();

        abstract Contribution calc();
    }

    private static class InitialContribution extends Contribution {

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

    private static class SumContribution extends Contribution {

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
        if (lastContribution == null && model instanceof ComputationGraph) {
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
                    .addListener(epsilon -> lastContribution.setEps(epsilon.detach()));
        } else {
            throw new IllegalArgumentException("Can only work on ComputationGraph instances! Got " + model);
        }
        lastContribution.setAct(activations.get(layerName).detach());
    }


    @Override
    public void onGradientCalculation(Model model) {
        try {
            lastContribution = lastContribution.calc();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to calculate contribution for layer " + layerName, e);
        }


        // Most likely not useable, but keep at least one version of it in history....
//        final String weightName = layerName + "_" + DefaultParamInitializer.WEIGHT_KEY;
//        final String biasName = layerName + "_" + DefaultParamInitializer.BIAS_KEY;
//        final INDArray weightGrad = model.gradient().getGradientFor(weightName);
//        final INDArray biasGrad = Optional.ofNullable(model.gradient().getGradientFor(biasName))
//                .orElse(Nd4j.zeros(1, weightGrad.size(1)));
//
//        final long batchSize = lastActivation.size(0); // Assume 0 is batchdimension
//
//        if(weightGrad.rank() == 2) {
//        final INDArray contribution = weightGrad.add(biasGrad.getColumn(0).broadcast(weightGrad.shape())).mul(lastActivation)
//                .div(batchSize).mean(1);
//        } else if(weightGrad.rank() == 4){
//            final INDArray contribution = weightGrad.amean(1, 2, 3).swapAxes(0,1).add(biasGrad)
//                    .mul(lastActivation.amean(0, 2, 3));
//        }
    }

    @Override
    public void onEpochEnd(Model model) {
       listener.accept(lastContribution.getContrib());
       lastContribution = new InitialContribution();
    }
}
