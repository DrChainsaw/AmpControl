package ampcontrol.model.training.data.state;

import java.util.Random;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**
 * Simple state factory
 *
 * @author Christian Skärby
 */
public class SimpleStateFactory implements StateFactory {

    private long nextSeed;

    /**
     * Constructor
     * @param nextSeed seed for the first random
     */
    public SimpleStateFactory(long nextSeed) {
        this.nextSeed = nextSeed;
    }

    @Override
    public Random createNewRandom() {
        return new Random(nextSeed);
    }

    @Override
    public <T> Supplier<T> createNewStateReference(UnaryOperator<T> storeFunction, T initial) {
        return () -> initial;
    }
}
