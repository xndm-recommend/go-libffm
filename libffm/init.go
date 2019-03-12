package libffm

import (
	"fmt"
	"os"

	"github.com/xndm-recommend/go-utils/errors_"
)

type LibFFMOptions struct {
	Model_path string // 模型路径
}

type ffm_node struct {
	F int     // field index
	J int     // feature index
	V float32 // value
}

type LibFFmClient struct {
	Model_path string
	ModelData  *ffm_model
}

func pathExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

func (c *LibFFmClient) readModel(model_path string) error {
	var b bool
	var err error
	if b, err = pathExists(model_path); true == b && nil == err {
		if dataTmp, b := loadFfmModel(model_path); true == b {
			c.ModelData = &dataTmp
		} else {
			return fmt.Errorf("libffm model read error!!!")
		}
	}
	return err
}

func (c *LibFFmClient) ReloadModel(model string) error {
	c.Model_path = model
	return c.readModel(c.Model_path)
}

func (c *LibFFmClient) init() {
	// 模型初始化
	err := c.readModel(c.Model_path)
	errors_.CheckFatalErr(err)
}

func NewLibFFMClient(opt *LibFFMOptions) (*LibFFmClient, error) {
	c := &LibFFmClient{
		Model_path: opt.Model_path,
		ModelData:  new(ffm_model),
	}
	c.init()
	if false == c.ModelData.normalization {
		return c, fmt.Errorf("libffm model read error!!!", c.ModelData)
	}
	return c, nil
}
