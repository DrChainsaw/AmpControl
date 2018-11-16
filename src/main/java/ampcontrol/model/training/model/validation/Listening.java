package ampcontrol.model.training.model.validation;

import org.deeplearning4j.eval.IEvaluation;

import java.util.Optional;
import java.util.function.Consumer;

public class Listening<T extends IEvaluation> implements Validation<T> {

    private final Validation<T> sourceValidation;
    private final Consumer<Boolean> listener;

    public Listening(Consumer<Boolean> listener, Validation<T> sourceValidation) {
        this.sourceValidation = sourceValidation;
        this.listener = listener;
    }

    @Override
    public Optional<T> get() {
        final Optional<T> optIEval = sourceValidation.get();
        listener.accept(optIEval.isPresent());
        return optIEval;
    }

    @Override
    public void notifyComplete() {
        sourceValidation.notifyComplete();
    }
}
