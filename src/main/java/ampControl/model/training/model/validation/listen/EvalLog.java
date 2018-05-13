package ampControl.model.training.model.validation.listen;

import org.deeplearning4j.eval.Evaluation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.Consumer;

/**
 * Logs evaluation results
 *
 * @author Christian Skärby
 */
public class EvalLog implements Consumer<Evaluation> {
    private static final Logger log = LoggerFactory.getLogger(EvalLog.class);

    private final String modelName;

    private double bestAccuracy;

    /**
     * Constructor
     * @param modelName name of model for which evaluation is provided
     */
    public EvalLog(String modelName, double bestAccuracy) {
        this.modelName = modelName;
        this.bestAccuracy = bestAccuracy;
    }

    @Override
    public void accept(Evaluation eval) {
        final double newAccuracy = eval.accuracy();
        bestAccuracy = Math.max(bestAccuracy, newAccuracy);
        log.info("Eval report for " + modelName);
        log.info(eval.stats());
        log.info("\n" + eval.confusionToString());
        log.info("Accuracy = " + newAccuracy + " Best: " + bestAccuracy);
    }
}
