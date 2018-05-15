package ampControl.model.training.model.validation.listen;

import org.deeplearning4j.eval.Evaluation;

import java.util.function.Predicate;

/**
 * Returns true accuracy has improved over last best accuracy. Possible to set a threshold relative to best for test
 * to pass.
 *
 * @author Christian Skärby
 */
public class AccuracyImproved implements Predicate<Evaluation> {
    private double bestAccuracy;
    private final double threshold;

    /**
     * Constructor
     * @param initialBestAccuracy initial best accuracy
     */
    public AccuracyImproved(double initialBestAccuracy) {
        this(initialBestAccuracy, 1);
    }

    /**
     * Constructor
     * @param initialBestAccuracy initial best accuracy
     * @param threshold threshold vs accuracy for test to pass
     */
    public AccuracyImproved(double initialBestAccuracy, double threshold) {
        this.bestAccuracy = initialBestAccuracy;
        this.threshold = threshold;
    }

    @Override
    public boolean test(Evaluation eval) {
        boolean ok = eval.accuracy() >= threshold * bestAccuracy;
        bestAccuracy = Math.max(eval.accuracy(), bestAccuracy);
        return ok;
    }
}
