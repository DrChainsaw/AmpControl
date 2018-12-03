package ampcontrol.model.training.model.validation.listen;

import org.junit.Test;
import org.nd4j.evaluation.classification.Evaluation;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link BestEvalScore}
 *
 * @author Christian Skärby
 */
public class BestEvalScoreTest {

    /**
     * Test that correct score is provided
     */
    @Test
    public void testScoreUpdate() {
        try {
            final BestEvalScore score = new BestEvalScore("thjtrfwaw");
            double expected = 0;
            assertEquals("Incorrect score!", expected , score.get(), 1e-10);
            final Evaluation eval = new Evaluation(2);
            eval.eval(0,1);
            eval.eval(1,1);

            expected = eval.accuracy();
            score.accept(eval);
            assertEquals("Incorrect score!", expected , score.get(), 1e-10);

            eval.eval(0,1);
            score.accept(eval);
            assertEquals("Incorrect score!", expected , score.get(), 1e-10);

            eval.eval(1,1);
            eval.eval(1,1);
            expected = eval.accuracy();
            score.accept(eval);
            assertEquals("Incorrect score!", expected , score.get(), 1e-10);



        } catch (IOException e) {
            throw new IllegalStateException("Failed to create instance!", e);
        }


    }
}