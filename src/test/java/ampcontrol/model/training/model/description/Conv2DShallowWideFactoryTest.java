package ampcontrol.model.training.model.description;

import org.junit.Test;

import java.util.ArrayList;

/**
 * Test cases for {@link Conv2DShallowWideFactory}
 *
 * @author Christian Skärby
 */
public class Conv2DShallowWideFactoryTest {

    /**
     * Just a smoke test...
     */
    @Test
    public void addModelData() {
        new FactoryInitializer().initialize(
                ((tr, ev, is, np, md) -> new Conv2DShallowWideFactory(tr,ev,is,np,md).addModelData(new ArrayList<>()))
        );
    }
}