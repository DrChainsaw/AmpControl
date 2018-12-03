package ampcontrol.model.training.model.naming;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link AddPrefix}
 */
public class AddPrefixTest {

    /**
     * Test simple prefix added
     */
    @Test
    public void toFileNameTransparent() {
        final String prefix = "prefix_";
        final String name = "name";
        assertEquals("Incorrect name!", prefix + name, new AddPrefix(prefix).toFileName(name));
    }
}