package ampcontrol.model.training.model.validation.listen;

import org.junit.Test;

import java.util.function.Consumer;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link Gated}
 *
 * @author Christian Skärby
 */
public class GatedTest {

    /**
     * Test gate open
     */
    @Test
    public void gateOpen() {
        final ProbeConsumer<Boolean> probe = new ProbeConsumer<>();
        new Gated<>(probe, t -> true).accept(true);
        probe.assertWasCalled(true);
    }

    /**
     * Test gate closed
     */
    @Test
    public void gateClosed() {
        final ProbeConsumer<Boolean> probe = new ProbeConsumer<>();
        new Gated<>(probe, t -> false).accept(true);
        probe.assertWasCalled(null);
    }

    public static class ProbeConsumer<T> implements Consumer<T> {

        private T last;

        @Override
        public void accept(T t) {
            last = t;
        }

        public void assertWasCalled(T expected) {
            assertEquals("Incorrect consumable! ", expected, last);
        }
    }
}