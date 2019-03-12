package main

import (
	"fmt"
	"go-libffm/libffm"
	"time"
)

func main() {
	var ffm_predict_file []string = []string{"1 1:227:0.062500 1:381:0.031250 1:382:0.031250 " +
		"1:590:0.125000 1:610:0.062500 1:594:0.031250 1:588:0.031250 1:386:0.062500 " +
		"2:1162:1 3:902:1 4:198:0.890000 5:1264:1 6:618:1 6:622:1 6:1411:1"}
	c, err := libffm.NewLibFFMClient(&libffm.LibFFMOptions{
		Model_path: "model/model_ffm.out",
	})
	fmt.Println(err)
	start := time.Now()
	f_out, _, err := c.LibFFM_Predict_apply(ffm_predict_file)
	fmt.Println("f_out", f_out)
	cost := time.Since(start)
	fmt.Println("cost=", cost)
}
