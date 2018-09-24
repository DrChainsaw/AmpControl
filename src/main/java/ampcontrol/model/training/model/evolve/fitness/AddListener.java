package ampcontrol.model.training.model.evolve.fitness;

import ampcontrol.model.training.model.ModelAdapter;
import org.deeplearning4j.optimize.api.TrainingListener;

import java.util.function.Consumer;
import java.util.function.Function;

/**
 * {@link FitnessPolicy} which only appends listeners to candidates. Basically a kind of API abuse since this class
 * will be called once for every candidate in a generation when the generation is created.
 * @param <T>
 *
 * @author Christian Skärby
 */
public class AddListener<T extends ModelAdapter> implements FitnessPolicy<T> {

    private final Function<Consumer<Double>, TrainingListener> listenerSupplier;

    public AddListener(Function<Consumer<Double>, TrainingListener>  listenerSupplier) {
        this.listenerSupplier = listenerSupplier;
    }

    @Override
    public T apply(T candidate, Consumer<Double> fitnessListener) {
        candidate.asModel().addListeners(listenerSupplier.apply(fitnessListener));
        return candidate;
    }
}
