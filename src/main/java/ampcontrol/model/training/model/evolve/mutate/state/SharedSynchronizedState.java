package ampcontrol.model.training.model.evolve.mutate.state;

import java.util.ArrayList;
import java.util.List;

/**
 * State which may be shared across multiple independent {@link View} instances. Each {@link View}will be notified when
 * the state changes.
 * <br><br>
 * A {@link View} can also be copied to create a new independent instance with the same (initial) state. Updating the
 * state of the new {@link View} will not affect the state of the old and vice versa.
 * <br><br>
 * This allows for state to stay consistent while being updated and copied independently across different views.
 * <br><br>
 * Word of warning: This is not foolproof! Interleaving state update and copying in different contexts is not guaranteed
 * to produce wanted results. For example:
 * <pre>
 *     ContextB.view1 = ContextA.view1.copy()
 *     ContextB.view1.update(newState)
 *     ContextB.view2 = ContextA.view2.copy()
 * </pre>
 * The above will result in state of ContextB.view1, ContextB.view2 <i>and</i> ContextA.view2 having state
 * "{@code newState}" after sequence is completed while ContextA.view1 will have the old state (assuming view1 and view2
 * are {@code View}s on the same {@code SharedSynchronizedState}. In practice this means that all views in a context
 * must be copied before any state is updated for that context to stay consistent.
 *
 * @author Christian Skärby
 */
public final class SharedSynchronizedState<T> {

    private final T state;
    private final List<View<T>> views;

    public static final class View<T> {
        private SharedSynchronizedState<T> state;

        private View() {
            // Shall only be created internally
        }

        private void set(SharedSynchronizedState<T> state)  {
            this.state = state;
        }

        /**
         * Get the current state
         * @return the current state
         */
        public T get() {
            return state.get();
        }

        /**
         * Update the current state
         * @param newState the new state
         */
        public View<T> update(T newState) {
            state.updateState(newState);
            return this;
        }

        /**
         * Copy the current state into a new independent instance
         * @return a new View on the copied state
         */
        public View<T> copy() {
            final View<T> newView = new View<>();
            state.updateListener(this, newView);
            return newView;
        }

    }

    /**
     * Constructor
     * @param state State to be shared across {@link View}s
     */
    public SharedSynchronizedState(T state) {
        this(state, new ArrayList<>());
    }

    /**
     * Internal constructor for when state or views change
     * @param state State to be shared across {@link View}s
     * @param views {@link View}s of the shared state
     */
    private SharedSynchronizedState(T state, List<View<T>> views) {
        this.state = state;
        this.views = views;
    }

    /**
     * Create a new {@link View} instance of this {@link SharedSynchronizedState}. View will be synchronized to all
     * views on the current state.
     * @return a new {@link View}.
     */
    public View<T> view() {
        final View<T> newView = new View<>();
        newView.set(this);
        views.add(newView);
        return newView;
    }

    /**
     * Update the shared state
     * @param newState new state
     */
    private void updateState(T newState) {
        final SharedSynchronizedState<T> updatedState = new SharedSynchronizedState<>(newState, views);
        views.forEach(listener -> listener.set(updatedState));
    }

    /**
     * Update a listener to a new reference. This is basically the mechanism for copying the state through independent
     * calls
     * @param toRemove View to remove
     * @param toAdd View to add
     */
    private void updateListener(View<T> toRemove, View<T> toAdd) {
        final List<View<T>> newViews = new ArrayList<>(views);
        if(!newViews.remove(toRemove)) {
            throw new IllegalArgumentException("Tried to update listener reference which does not exist!");
        }
        newViews.add(toAdd);
        final SharedSynchronizedState<T> updatedState = new SharedSynchronizedState<>(state, newViews);
        newViews.forEach(listener -> listener.set(updatedState));
    }

    /**
     * Return the state
     * @return the state
     */
    private T get() {
        return state;
    }
}
