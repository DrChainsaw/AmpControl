package ampcontrol.model.inference;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Does Exponential Weighted Moving Averqage in time of the classifications of the source {@link Classifier}. The
 * forgettingFactor sets how much of the past to forget, effectively setting the tradeoff between noisyness and delay.
 *
 * @author Christian Skärby
 */
class EwmaClassifier implements Classifier {

    private final double ff;
    private final double ff_1;
    private final Classifier sourceClassifier;
    private INDArray state;

    /**
     * Constructor
     * @param forgettingFactor Sets filter constant of EWMA filter. {@code Y[n] = forgettingFactor*X[n] + (1-forgettingFactor)*Y[n-1]}
     * @param sourceClassifier {@link Classifier} to be filtered
     */
    EwmaClassifier(double forgettingFactor, Classifier sourceClassifier) {
        this.ff= forgettingFactor;
        ff_1 = 1 - forgettingFactor;
        this.sourceClassifier = sourceClassifier;
    }

    @Override
    public INDArray classify() {
        if(state == null) {
            state = sourceClassifier.classify();
        } else {
            state.muli(ff_1).addi(sourceClassifier.classify().mul(ff));
        }
        return state;
    }

	@Override
	public double getAccuracy() {
		return sourceClassifier.getAccuracy();
	}
}
