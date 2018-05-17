package ampcontrol.model.training.model.description;

import org.junit.Test;

import java.util.ArrayList;

/**
 * Test cases for {@link ResNetConv2DFactory}
 *
 * @author Christian Skärby
 */
public class StackedConv2DFactoryTest {

    /**
     * Just a smoke test...
     */
    @Test
    public void addModelData() {
        new FactoryInitializer()
                .initialize(
                        ((tr, ev, is, np, md) -> new StackedConv2DFactory(tr,ev,is,np,md).addModelData(new ArrayList<>()))
                );
    }
}