package ampcontrol.model.training.model.validation;

import org.deeplearning4j.eval.Evaluation;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link CachingValidationFactory}
 *
 * @author Christian Skärby
 */
public class CachingValidationFactoryTest {

    /**
     * Test that create method produces the same instance when called twice
     */
    @Test
    public void create() {
        final Validation.Factory<Evaluation> factory = (labs) -> new EvalValidation(new Evaluation(labs), (eval) -> {
        });
        final CachingValidationFactory<Evaluation> cachingFactory = new CachingValidationFactory<>(factory);
        final List<String> labels = Arrays.asList("labels".split("(?!^)"));
        assertEquals("Same instance expected!", cachingFactory.create(labels), cachingFactory.create(labels));
    }

    /**
     * Test that create method produces the same instance when called twice
     */
    @Test(expected = IllegalArgumentException.class)
    public void createWrongLabes() {
        final Validation.Factory<Evaluation> factory = (labs) -> new EvalValidation(new Evaluation(labs), (eval) -> {
        });
        final CachingValidationFactory<Evaluation> cachingFactory = new CachingValidationFactory<>(factory);
        final List<String> labels = Arrays.asList("labels".split("(?!^)"));
        cachingFactory.create(labels);
        cachingFactory.create(labels.subList(1, labels.size()-1));
    }
}