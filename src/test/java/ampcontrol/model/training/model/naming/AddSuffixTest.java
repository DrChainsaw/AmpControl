package ampcontrol.model.training.model.naming;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link AddSuffix}
 * 
 * @author Christian Skärby
 */
public class AddSuffixTest {

    /**
     * Test simple suffix added
     */
    @Test
    public void toFileNameTransparent() {
        final String suffix = "_suffix";
        final String name = "name";
        assertEquals("Incorrect name!", name + suffix, new AddSuffix(suffix).toFileName(name));
    }
}