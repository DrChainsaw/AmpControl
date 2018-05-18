package ampcontrol.model.training.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

/**
 * {@link Function} which shuffles the provided {@link List} without affecting the original.
 * @param <T> The type of elements in the list
 *
 * @author Christian Skärby
 */
public class ListShuffler<T> implements Function<List<T>, List<T>> {

    private final Random rng;

    public ListShuffler(Random rng) {
        this.rng = rng;
        rng.nextInt();
        rng.nextInt();
        rng.nextInt();
    }

    @Override
    public List<T> apply(List<T> ts) {
        List<T> toRet = new ArrayList<>(ts);
        Collections.shuffle(toRet, rng);
        return toRet;
    }
}
