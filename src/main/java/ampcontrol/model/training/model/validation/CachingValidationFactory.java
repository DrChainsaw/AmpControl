package ampcontrol.model.training.model.validation;

import org.deeplearning4j.eval.IEvaluation;

import java.util.ArrayList;
import java.util.List;

/**
 * {@link Validation.Factory} which stores the created instance so that a new one is not created
 * @param <T>
 *
 * @author Christian Skärby
 */
public final class CachingValidationFactory<T extends IEvaluation> implements Validation.Factory<T> {

    private final Validation.Factory<T> sourceFactory;
    private Validation<T> validation;
    private final List<String> labels = new ArrayList<>();

    public CachingValidationFactory(Validation.Factory<T> sourceFactory) {
        this.sourceFactory = sourceFactory;
    }

    @Override
    public Validation<T> create(List<String> labels) {
        if (validation == null) {
            validation = sourceFactory.create(labels);
            this.labels.addAll(labels);
        }
        if(!this.labels.equals(labels)){
            throw new IllegalArgumentException("Incorrect labels! Cache created for " + this.labels + " got " + labels);
        }

        return validation;
    }
}
