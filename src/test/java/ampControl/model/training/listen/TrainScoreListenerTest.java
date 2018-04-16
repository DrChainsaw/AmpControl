package ampControl.model.training.listen;

import org.junit.Test;

import java.util.function.BiConsumer;
import java.util.stream.DoubleStream;

import static org.junit.Assert.*;

/**
 * Test cases for {@link TrainScoreListener}
 *
 * @author Christian Skärby
 */
public class TrainScoreListenerTest {

    /**
     * Test iterationDone
     */
    @Test
    public void iterationDone() {
        final int reportInterval = 2;
        final double score0 = 666;
        final int iter0 = 0;
        final double score1 = 333;
        final int iter1 = reportInterval-1;

        final double score2 = 111;
        final int iter2 = reportInterval;

        final ProbeConsumer probe = new ProbeConsumer();
        final TrainScoreListener listener = new TrainScoreListener(reportInterval, probe);

        listener.iterationDone(new ScoreModel(score0), iter0,0);
        probe.assertValues(iter0, score0);

        listener.iterationDone(new ScoreModel(score1), iter1,0);
        probe.assertValues(iter0, score0);

        listener.iterationDone(new ScoreModel(score2), iter2,0);
        final double expectedScore = DoubleStream.of(score1, score2).summaryStatistics().getAverage();
        probe.assertValues(iter2, expectedScore);

        listener.iterationDone(new ScoreModel(score0), iter0,0);
        probe.assertValues(iter2, expectedScore);

        listener.iterationDone(new ScoreModel(score1), iter1,0);
        probe.assertValues(iter1, DoubleStream.of(score0, score1).summaryStatistics().getAverage());

    }

    private static class ProbeConsumer implements BiConsumer<Integer, Double> {

        private int lastInt = -1;
        private double lastDouble = Double.NaN;

        @Override
        public void accept(Integer integer, Double aDouble) {
            lastInt = integer;
            lastDouble = aDouble;
        }

        private void assertValues(int expectedIter, double expectedScore) {
            assertEquals("Incorrect iteration!", expectedIter, lastInt);
            assertEquals("Incorrect score!", expectedScore, lastDouble, 1e-10);

        }
    }
}