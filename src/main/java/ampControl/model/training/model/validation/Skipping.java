package ampControl.model.training.model.validation;

import org.deeplearning4j.eval.IEvaluation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.function.Function;

/**
 * Allows for skipping a number future validations based on characteristics of a produced {@link IEvaluation}. Intended
 * use is to validate models more often when accuracy gets better.
 *
 * @author Christian Skärby
 */
public class Skipping<T extends IEvaluation> implements Validation<T> {

    private static final Logger log = LoggerFactory.getLogger(Skipping.class);

    private final Validation<T> sourceValidation;
    private final Function<T, Integer> metricToNrToSkip;

    private int nrofValidationsToSkip;
    private Optional<T> last = Optional.empty();

    public Skipping(Function<T, Integer> metricToNrToSkip, Validation<T> sourceValidation) {
        this(metricToNrToSkip, 0, sourceValidation);
    }

    public Skipping(Function<T, Integer> metricToNrToSkip, int intialNrToSkip, Validation<T> sourceValidation) {
        this.sourceValidation = sourceValidation;
        this.metricToNrToSkip = metricToNrToSkip;
        this.nrofValidationsToSkip = intialNrToSkip;
    }

    @Override
    public Optional<T> get() {
        if(nrofValidationsToSkip > 0) {
            nrofValidationsToSkip--;
            log.info("Skip eval! " + nrofValidationsToSkip);
            return Optional.empty();
        }
        last = sourceValidation.get();
        return last;
    }

    @Override
    public void notifyComplete() {
        nrofValidationsToSkip = last.map(metricToNrToSkip).orElse(nrofValidationsToSkip);
        last.ifPresent(dummy -> sourceValidation.notifyComplete());
    }
}
