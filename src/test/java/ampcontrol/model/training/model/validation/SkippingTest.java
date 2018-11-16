package ampcontrol.model.training.model.validation;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.IEvaluation;
import org.junit.Test;

import java.util.Optional;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Test cases for {@link Skipping}.
 *
 * @author Christian Skärby
 */
public class SkippingTest {

    /**
     * Test no skipping
     */
    @Test
    public void testNoSkip() {
        final ProbeValidation<Evaluation> probe = new ProbeValidation<>(new Evaluation());
        final Skipping<Evaluation> skipping = new Skipping<>(t -> 0, probe);
        assertTrue("Expect to get something!!", skipping.get().isPresent());
        skipping.notifyComplete();
        assertTrue("Expect to get something!!", skipping.get().isPresent());
        probe.assertNrofNotifications(1);
    }

    /**
     * Test skipping
     */
    @Test
    public void testSkip() {
        final int nrToSkip = 2;
        final ProbeValidation<Evaluation> probe = new ProbeValidation<>(new Evaluation());
        final Skipping<Evaluation> skipping = new Skipping<>(t -> nrToSkip, probe);
        assertTrue("Expect to get something!!", skipping.get().isPresent());
        skipping.notifyComplete();
        IntStream.range(1, nrToSkip + 1)
                .forEach(i -> assertFalse("Did not expect to get something after " + i + " tries!!", skipping.get().isPresent()));
        assertTrue("Expect to get something!!", skipping.get().isPresent());
        skipping.notifyComplete();
        probe.assertNrofNotifications(2);
    }

    private static class ProbeValidation<T extends IEvaluation> implements Validation<T> {

        private final T t;
        private int nrofNotify = 0;

        public ProbeValidation(T t) {
            this.t = t;
        }

        @Override
        public Optional<T> get() {
            return Optional.of(t);
        }

        @Override
        public void notifyComplete() {
            nrofNotify++;
        }

        private void assertNrofNotifications(int expected) {
            assertEquals("Incorrect number of notifications!", expected, nrofNotify);
        }
    }
}