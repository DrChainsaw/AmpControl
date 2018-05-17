package ampcontrol.model.training.data;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link MultiplyLabelExpander}
 *
 * @author Christian Skärby
 */
public class MultiplyLabelExpanderTest {

    /**
     * Test apply
     */
    @Test
    public void apply() {
        final String remove = "aa";
        final String copyTwice = "bb";
        final String dontTouch = "cc";
        final List<String> testList = Arrays.asList(remove, copyTwice, dontTouch, copyTwice, remove);
        final List<String> expected = Arrays.asList(copyTwice, copyTwice, dontTouch, copyTwice, copyTwice);
        final List<String> result = new MultiplyLabelExpander()
                .addExpansion(remove,0)
                .addExpansion(copyTwice, 2)
                .apply(testList);
        assertEquals("Incorrect result!", expected, result);
    }
}