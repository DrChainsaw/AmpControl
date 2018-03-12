package ampControl.audio.processing;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Test cases for {@link MaskBins}
 *
 * @author Christian Skärby
 */
public class MaskBinsTest {

    /**
     * Test that bins are masked out
     */
    @Test
    public void receive() {
        final double[][] test =     {{1,2,3}, {4,5,6}};
        final double[][] expected = {{0,2,0}, {0,5,0}};
        final int[] toMask = {0, 2};
        final ProcessingResult.Processing mask = new MaskBins(toMask);
        mask.receive(test);
        assertArrayEquals("Masking was not applied correctly", expected, mask.get().get(0));
    }

    /**
     * Test that name is consistent and can be used to create a new instance
     */
    @Test
    public void name() {
        ProcessingResult.Processing maskExpected = new MaskBins(new int[] {0,2,4,6});
        assertEquals("Incorrect instance!", new MaskBins(maskExpected.name()).name(), maskExpected.name());
    }

    /**
     * Test that name is consistent with static version
     */
    @Test
    public void nameStatic() {
        assertEquals("Incorrect instance!", MaskBins.nameStatic(), new MaskBins(new int[0]).name());
    }
}