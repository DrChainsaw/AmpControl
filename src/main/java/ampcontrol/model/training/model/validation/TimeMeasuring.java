package ampcontrol.model.training.model.validation;

import org.deeplearning4j.eval.Evaluation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.LocalTime;
import java.util.Optional;

/**
 * Adds time measurement to an validation. Only measure time when source wants to validate.
 *
 * @author Christian Skärby
 */
public class TimeMeasuring implements Validation<Evaluation> {

    private static final Logger log = LoggerFactory.getLogger(TimeMeasuring.class);

    private final Validation<Evaluation> sourceValidation;

    private LocalTime startTime;
    private Optional<Evaluation> last = Optional.empty();

    /**
     * Constructor
     * @param sourceValidation Validation to use
     */
    public TimeMeasuring(Validation<Evaluation> sourceValidation) {
        this.sourceValidation = sourceValidation;
    }

    @Override
    public Optional<Evaluation> get() {
        last = sourceValidation.get();
        last.ifPresent(eval -> startTime = LocalTime.now());
        return last;
    }

    @Override
    public void notifyComplete() {
        last.ifPresent(eval -> {
            final double endTimeMs = Duration.between(startTime, LocalTime.now()).toMillis();
            log.info("Evaluation took " + endTimeMs + " ms for " + eval.getNumRowCounter() + " examples, " + endTimeMs / eval.getNumRowCounter() + " ms per example");
        });
        sourceValidation.notifyComplete();
    }
}
