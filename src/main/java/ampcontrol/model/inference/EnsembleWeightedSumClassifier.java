package ampcontrol.model.inference;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.OldSoftMax;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

/**
 * Ensemble classifier which sums the accuracy weighted probabilities from a list of {@link Classifier Classifiers}
 * and normalizes the result through the provided {@link BiFunction} which gets the weighted sum and the sum of the
 * individual classifiers as input.
 *
 * @author Christian Skärby
 */
class EnsembleWeightedSumClassifier implements Classifier {


    final static BiFunction<Double, INDArray, INDArray>  avgNormalizer = (sumAcc, aggClass) -> aggClass.div(sumAcc);
    final static BiFunction<Double, INDArray, INDArray> softMaxNormalizer =  (sumAcc, aggClass) -> Nd4j.getExecutioner().execAndReturn(new OldSoftMax(aggClass));

    private final List<Classifier> ensemble;
    private final BiFunction<Double, INDArray, INDArray> normalizer;
    private final double accuracy;

    /**
     * Constructor
     *
     * @param ensemble the ensemble for which classifications are combined
     * @param normalizer defines how to renormalize the weighted sum into a probability vector
     */
    EnsembleWeightedSumClassifier(List<? extends Classifier> ensemble,
                                         BiFunction<Double, INDArray, INDArray> normalizer) {
        this.ensemble = new ArrayList<>(ensemble);
        this.normalizer = normalizer;
        accuracy = ensemble.stream()
        		.mapToDouble(Classifier::getAccuracy)
        		.max()
        		.orElse(0);
    }

    @Override
    public INDArray classify() {
        INDArray ret = null;
        double sumAcc = 0;
        for(Classifier classifier: ensemble) {
        	sumAcc += classifier.getAccuracy();
            if(ret == null) {
                ret = classifier.classify().mul(classifier.getAccuracy());
            } else {
            	ret.addi(classifier.classify().mul(classifier.getAccuracy()));
            }
        }
        return normalizer.apply(sumAcc, ret);//ret.div(sumAcc);
    }

	@Override
	public double getAccuracy() {
		return accuracy;
	}

}
