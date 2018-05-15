package ampControl.model.training.model.validation.listen;

import org.deeplearning4j.eval.Evaluation;
import org.junit.Test;

import java.util.function.Predicate;

import static junit.framework.TestCase.assertFalse;
import static junit.framework.TestCase.assertTrue;

/**
 * Test cases for {@link AccuracyImproved}
 *
 * @author Christian Skärby
 */
public class AccuracyImprovedTest {

    /**
     * Test without threshold
     */
    @Test
    public void testNoThreshold() {
        final double initalAcc = 0.4;
        final Predicate<Evaluation> accuracyImproved = new AccuracyImproved(initalAcc);
        final Evaluation better = new Evaluation(2);
        better.eval(1,1);
        assertTrue("Accuracy shall be improved!", accuracyImproved.test(better));
        assertTrue("Accuracy shall be improved!", accuracyImproved.test(better));

        final Evaluation worse = new Evaluation(2);
        worse.eval(0,1);
        worse.eval(1,1);
        assertFalse("Accuracy was not improved!", accuracyImproved.test(worse));
    }

    /**
     * Test without threshold
     */
    @Test
    public void testWithThreshold() {
        final double initalAcc = 1;
        final double threshold = 0.4;
        final Predicate<Evaluation> accuracyImproved = new AccuracyImproved(initalAcc, threshold);

        final Evaluation better = new Evaluation(2);
        better.eval(0,1);
        better.eval(1,1);
        assertTrue("Accuracy shall be improved!", accuracyImproved.test(better));

    }
}