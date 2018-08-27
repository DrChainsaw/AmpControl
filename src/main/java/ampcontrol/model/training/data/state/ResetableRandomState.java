package ampcontrol.model.training.data.state;

import java.util.Random;

/**
 * {@link ResetableState} for {@link Random}. Since it is not possible to get the current state of {@link Random}
 * , this class stores the seed and rolls a new seed from the random when {@link #storeCurrentState()} is called.
 *
 * @author Christian Skärby
 */
public class ResetableRandomState implements ResetableState {

    private final Random rng;
    private long currentState;

    public ResetableRandomState(long seed) {
        currentState = seed;
        rng = new Random(seed);
    }

    /**
     * Returns the rng controlled by this class
     * @return the rng
     */
    public Random getRandom() {
        return rng;
    }

    @Override
    public void storeCurrentState() {
        currentState = rng.nextLong();
        restorePreviousState();
    }

    @Override
    public void restorePreviousState() {
        rng.setSeed(currentState);
    }
}
