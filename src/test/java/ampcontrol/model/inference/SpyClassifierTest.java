package ampcontrol.model.inference;

import ampcontrol.audio.ClassifierInputProvider;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link SpyClassifier}
 *
 * @author Christian Skärby
 */
public class SpyClassifierTest {

    /**
     * Test that reporting of correctly and incorrectly classified input works as expected
     */
    @Test
    public void spyInputTypes() {
        final int correctClassifictionInd = 0;
        final int incorrectClassifictionInd = 1;
        final INDArray correctClassifiction = Nd4j.zeros(2).putScalar(correctClassifictionInd, 1);
        final INDArray incorrectClassifiction = Nd4j.zeros(2).putScalar(incorrectClassifictionInd, 1);
        final int maxAccum = 3;
        final SpyClassifier.StoringListener listener = new SpyClassifier.StoringListener();
        final List<INDArray> classifications = Arrays.asList(
                correctClassifiction,   // 0
                incorrectClassifiction, // 1
                correctClassifiction,   // 2
                correctClassifiction,   // 3
                incorrectClassifiction, // 4
                incorrectClassifiction, // 5
                correctClassifiction,   // 6  -> expected
                correctClassifiction,   // 7  -> expected
                correctClassifiction,   // 8  -> expected
                incorrectClassifiction, // 9  -> expected
                incorrectClassifiction, // 10 -> expected
                incorrectClassifiction, // 11 -> expected
                incorrectClassifiction, // 12
                incorrectClassifiction  // 13
        );
        final INDArray expectedCorrect = Nd4j.create(new double[]{6, 7, 8});
        final INDArray expectedIncorrect = Nd4j.create(new double[]{9, 10, 11});
        final SpyClassifier spyClassifier = new SpyClassifier(
                new MockClassifier("mock", 0.7, classifications),
                new CountingInputProvider(),
                listener,
                maxAccum,
                correctClassifictionInd,
                incorrectClassifictionInd
        );

        for (int i = 0; i < classifications.size(); i++) {
            assertEquals("Not transparent!", classifications.get(i), spyClassifier.classify());
            assertEquals("Incorrect hasInput after " + i +" samples!", i >= 11, listener.hasInput());
        }

        assertEquals("Incorrect correctly classified input!", expectedCorrect, listener.getCorrectlyClassifiedInput());
        assertEquals("Incorrect incorrectly classified input!", expectedIncorrect, listener.getIncorrectlyClassifiedInput());
    }

    private static class CountingInputProvider implements ClassifierInputProvider {

        private int count = 0;

        @Override
        public INDArray getModelInput() {
            INDArray out = Nd4j.create(1, 1, 1, 1);
            out.putScalar(0,0,0,0, count++);
            return out;
        }
    }
}