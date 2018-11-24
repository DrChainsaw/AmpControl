package ampcontrol.model.training.model.evolve.mutate.state;

import ampcontrol.model.training.model.evolve.state.SharedSynchronizedState;
import ampcontrol.model.training.model.evolve.state.SharedSynchronizedState.View;
import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link SharedSynchronizedState}
 *
 * @author Christian Skärby
 */
public class SharedSynchronizedStateTest {

    /**
     * Test that the state is updated and shared correctly
     */
    @Test
    public void updateState() {
        final SharedSynchronizedState<String> initialState = new SharedSynchronizedState<>("First");

        final List<View<String>> viewsGen0 = Stream.generate(initialState::view).limit(5).collect(Collectors.toList());
        viewsGen0.forEach(view -> assertEquals("Incorrect state!", "First", view.get()));

        final List<View<String>> viewsGen1 = viewsGen0.stream().map(View::copy).collect(Collectors.toList());
        viewsGen1.forEach(view -> assertEquals("Incorrect state!", "First", view.get()));

        viewsGen1.get(0).update("Second");

        viewsGen0.forEach(view -> assertEquals("Incorrect state!", "First", view.get()));
        viewsGen1.forEach(view -> assertEquals("Incorrect state!", "Second", view.get()));

        viewsGen0.get(0).update("FirstNew");
        viewsGen0.forEach(view -> assertEquals("Incorrect state!", "FirstNew", view.get()));
        viewsGen1.forEach(view -> assertEquals("Incorrect state!", "Second", view.get()));
    }

    /**
     * Test that state can be corrupted. This is of course not a desired effect. Intention of testcase is to
     * show that effect exists and why.
     */
    @Test
    public void updateStateCorruption() {
        final SharedSynchronizedState<String> initialState = new SharedSynchronizedState<>("initialState");

        final View<String> view1 = initialState.view();
        final View<String> view2 = initialState.view();
        final View<String> copy = view1.copy();
        copy.update("newState");
        // This is in most cases not the desired outcome!
        assertEquals("Incorrect state!", "initialState", view1.get());
        assertEquals("Incorrect state!", "newState", view2.get());
        assertEquals("Incorrect state!", "newState", copy.get());
    }

}