package ampcontrol.model.training.model.evolve.mutate.util;

import ampcontrol.model.training.model.evolve.GraphUtils;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class CompGraphUtilTest {

    /**
     * Test two graphs with identical architecture get the same string while a third slightly different one gets a different string
     */
    @Test
    public void configUniquenessString() {
        final String conf1 = CompGraphUtil.configUniquenessString(
                GraphUtils.getForkResOuterInnerNet(
                        "0", "1",
                        new String[]{"f0_0", "f0_1"},
                        new String[]{"f1_0", "f1_1", "f1_2"}));

        final String conf2 = CompGraphUtil.configUniquenessString(
                GraphUtils.getForkResOuterInnerNet(
                        "0", "1",
                        new String[]{"f0_0", "f0_1"},
                        new String[]{"f1_0", "f1_1", "f1_2"}));

        final String conf3 = CompGraphUtil.configUniquenessString(
                GraphUtils.getForkResOuterInnerNet(
                        "0", "1",
                        new String[]{"f0_0", "f0__1"},
                        new String[]{"f1_0", "f1_1", "f1_2"}));

        System.out.println(conf1.length());

        assertEquals("Expected equal!", conf1, conf2);
        assertNotEquals("Expected not equal!", conf1, conf3);
    }
}