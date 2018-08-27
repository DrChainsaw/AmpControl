package ampcontrol.model.training.data.state;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**
 * {@link StateFactory} which is capable of storing and restoring all created state.
 * Wraps all created state in {@link ResetableState}s and stores a reference to the controllers in order to
 * do this.
 *
 * @author Christian Skärby
 */
public class ResetableStateFactory implements StateFactory, ResetableState {

    private final Collection<ResetableState> stateControllers = new ArrayList<>();

    private long nextSeed;

    /**
     * Constructor
     * @param nextSeed Next seed for any {@link Random} created
     */
    public ResetableStateFactory(long nextSeed) {
        this.nextSeed = nextSeed;
    }

    @Override
    public Random createNewRandom() {
        final ResetableRandomState controller = new ResetableRandomState(nextSeed++);
        stateControllers.add(controller);
        return controller.getRandom();
    }

    @Override
    public <T> Supplier<T> createNewStateReference(UnaryOperator<T> storeFunction, T initial) {
        final ResetableReferenceState controller = new ResetableReferenceState<>(storeFunction, initial);
        stateControllers.add(controller);
        return controller;
    }

    @Override
    public void storeCurrentState() {
        stateControllers.forEach(ResetableState::storeCurrentState);
    }

    @Override
    public void restorePreviousState() {
        stateControllers.forEach(ResetableState::restorePreviousState);
    }
}
