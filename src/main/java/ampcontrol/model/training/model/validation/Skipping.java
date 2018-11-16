package ampcontrol.model.training.model.validation;

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
    private final String logPrefix;

    private int nrofValidationsToSkip;
    private Optional<T> last = Optional.empty();

    Skipping(Function<T, Integer> metricToNrToSkip, Validation<T> sourceValidation) {
        this(metricToNrToSkip, 0, "", sourceValidation);
    }

    public Skipping(Function<T, Integer> metricToNrToSkip, String logPrefix, Validation<T> sourceValidation) {
        this(metricToNrToSkip, 0, logPrefix, sourceValidation);
    }

    public Skipping(Function<T, Integer> metricToNrToSkip, int initialNrToSkip, Validation<T> sourceValidation) {
        this(metricToNrToSkip, initialNrToSkip, "", sourceValidation);
    }

    public Skipping(Function<T, Integer> metricToNrToSkip, int initialNrToSkip, String logPrefix, Validation<T> sourceValidation) {
        this.sourceValidation = sourceValidation;
        this.metricToNrToSkip = metricToNrToSkip;
        this.nrofValidationsToSkip = initialNrToSkip;
        this.logPrefix = logPrefix;
    }

    @Override
    public Optional<T> get() {
        if(nrofValidationsToSkip > 0) {
            if(!logPrefix.isEmpty()) {
                log.info(logPrefix + nrofValidationsToSkip);
            }
            nrofValidationsToSkip--;
            last = Optional.empty();
        } else {
            last = sourceValidation.get();
        }
        return last;
    }

    @Override
    public void notifyComplete() {
        nrofValidationsToSkip = last.map(metricToNrToSkip).orElse(nrofValidationsToSkip);
        last.ifPresent(dummy -> sourceValidation.notifyComplete());
    }
}
