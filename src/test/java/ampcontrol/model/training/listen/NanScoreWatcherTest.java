package ampcontrol.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link NanScoreWatcher}
 *
 * @author Christian Skärby
 */
public class NanScoreWatcherTest {

    /**
     * Test iterationDone with a model having finite score
     */
    @Test
    public void iterationDoneFinite() {
        final Model finiteScoreModel = new ScoreModel(666);
        final ProbeCallback callback = new ProbeCallback();
        final NanScoreWatcher watcher = new NanScoreWatcher(callback);

        watcher.iterationDone(finiteScoreModel, 1,1);
        callback.assertWasCalled(false);
    }

    /**
     * Test iterationDone with a model having positive infinite score
     */
    @Test
    public void iterationDonePosInfinite() {
        final Model posInifinteScoreModel = new ScoreModel(Double.POSITIVE_INFINITY);
        final ProbeCallback callback = new ProbeCallback();
        final NanScoreWatcher watcher = new NanScoreWatcher(callback);

        watcher.iterationDone(posInifinteScoreModel, 1,1);
        callback.assertWasCalled(true);
    }

    /**
     * Test iterationDone with a model having negative infinite score
     */
    @Test
    public void iterationDoneNegInfinite() {
        final Model negInifinteScoreModel = new ScoreModel(Double.NEGATIVE_INFINITY);
        final ProbeCallback callback = new ProbeCallback();
        final NanScoreWatcher watcher = new NanScoreWatcher(callback);

        watcher.iterationDone(negInifinteScoreModel, 1,1);
        callback.assertWasCalled(true);
    }

    /**
     * Test iterationDone with a model having score NaN
     */
    @Test
    public void iterationDoneNan() {
        final Model nanScoreModel = new ScoreModel(Double.NaN);
        final ProbeCallback callback = new ProbeCallback();
        final NanScoreWatcher watcher = new NanScoreWatcher(callback);

        watcher.iterationDone(nanScoreModel, 1,1);
        callback.assertWasCalled(true);
    }

    private final static class ProbeCallback implements Runnable {

        private boolean wasCalled = false;
        @Override
        public void run() {
            wasCalled = true;
        }

        private void assertWasCalled(boolean expected) {
            assertEquals("Incorrect state!", expected, wasCalled);
        }
    }

}