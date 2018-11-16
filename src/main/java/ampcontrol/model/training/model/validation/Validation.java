package ampcontrol.model.training.model.validation;

import org.deeplearning4j.eval.IEvaluation;

import java.util.List;
import java.util.Optional;

/**
 * Model validation interface
 *
 * @param <T>
 */
public interface Validation<T extends IEvaluation> {

    interface Factory<T extends IEvaluation> {
        Validation<T> create(List<String> labels);
    }

    /**
     * Returns an {@link IEvaluation} to be evaluated.
     *
     * @return an {@link IEvaluation} to be evaluated.
     */
    Optional<T> get();

    /**
     * Callback when model evaluation is complete.
     */
    void notifyComplete();

}
