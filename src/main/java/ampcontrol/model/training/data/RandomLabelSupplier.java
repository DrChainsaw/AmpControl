package ampcontrol.model.training.data;

import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

/**
 * Supplies a random element from the given list.
 * @param <T>
 *
 * @author Christian Skärby
 */
public class RandomLabelSupplier<T> implements Supplier<T> {
	
	private final List<T> labels;
	private final Random rng;

	/**
	 * Constructor
	 * @param labels The list of labels to draw from
	 * @param rng Random number generator
	 */
	public RandomLabelSupplier(List<T> labels, Random rng) {
		this.labels = labels;
		this.rng = rng;
	}

	@Override
	public T get() {
		return labels.get(rng.nextInt(labels.size()));
	}

}
