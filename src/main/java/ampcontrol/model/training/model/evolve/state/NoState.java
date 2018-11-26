package ampcontrol.model.training.model.evolve.state;

/**
 * {@link AccessibleState} which has no state.
 * @param <S>
 *
 * @author Christian Skärby
 */
public class NoState<S> implements AccessibleState<S> {

    private final S fakeState;

    public NoState() {
        this(null);
    }

    public NoState(S fakeState) {
        this.fakeState = fakeState;
    }

    @Override
    public S get() {
        return fakeState;
    }

    @Override
    public void save(String baseName) {
        //Ignore
    }

    @Override
    public AccessibleState<S> clone() {
        return new NoState<>(fakeState);
    }
}
