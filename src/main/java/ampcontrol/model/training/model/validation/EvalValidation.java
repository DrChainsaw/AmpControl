package ampcontrol.model.training.model.validation;

import org.deeplearning4j.eval.Evaluation;

import java.util.Optional;
import java.util.function.Consumer;

/**
 * {@link Validation} which has an {@link Evaluation} instance.
 *
 */
public class EvalValidation implements Validation<Evaluation> {

    private final Evaluation evaluation;
    private final Consumer<Evaluation> listener;

    /**
     * Constructor
     * @param evaluation evaluation instance to use
     * @param listener listens to evaluation
     */
    public EvalValidation(Evaluation evaluation, Consumer<Evaluation> listener) {
        this.evaluation = evaluation;
        this.listener = listener;
    }

    @Override
    public Optional<Evaluation> get() {
        evaluation.reset();
        return Optional.of(evaluation);
    }

    @Override
    public void notifyComplete() {
        listener.accept(evaluation);
    }

}
