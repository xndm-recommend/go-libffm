package libffm

import (
	"math"
	"strconv"
	"strings"
)

func sigmod(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func wTx(ffm_node []ffm_node, model ffm_model, r, kappa, eta, lambda float32) float64 {
	var align0 int = 2 * get_k_aligned(model.k)
	var align1 int = model.m * align0
	var t float32 = 0
	for i, node := range ffm_node {
		j1 := node.J
		f1 := node.F
		v1 := node.V
		if j1 >= model.n || f1 >= model.m {
			continue
		}
		for j := i + 1; j < len(ffm_node); j++ {
			j2 := ffm_node[j].J
			f2 := ffm_node[j].F
			v2 := ffm_node[j].V
			if j2 >= model.n || f2 >= model.m {
				continue
			}
			v := v1 * v2 * r
			w1 := j1*align1 + f2*align0
			w2 := j2*align1 + f1*align0
			for d := 0; d < align0; d += int(iKALIGN * 2) {
				t += model.w[d+w1] * model.w[d+w2] * v
				t += model.w[d+w1+1] * model.w[d+w2+1] * v
				t += model.w[d+w1+2] * model.w[d+w2+2] * v
				t += model.w[d+w1+3] * model.w[d+w2+3] * v
			}
		}
	}
	return float64(t)
}

func ffm_predict(ffm_node []ffm_node, model ffm_model) float64 {
	var r float32 = 1
	if model.normalization {
		r = 0
		for _, node := range ffm_node {
			r += node.V * node.V
		}
		r = 1 / r
	}
	var kappa, eta, lambda float32 = 0, 0, 0
	t := wTx(ffm_node, model, r, kappa, eta, lambda)
	return sigmod(t)
}

func (c *LibFFmClient) LibFFM_Predict_apply(ffm_predict_data []string) ([]float64, float64, error) {
	var loss float64 = 0
	var f_out []float64
	var err error
	for _, predict := range ffm_predict_data {
		var y, _f float64
		var n ffm_node
		var x []ffm_node
		predict_list := strings.Split(predict, " ")
		if _int, _ := strconv.Atoi(predict_list[0]); _int > 0 {
			y = 1.0
		} else {
			y = -1.0
		}
		for _, s := range predict_list[1:] {
			y_char := strings.Split(s, ":")
			n.F, err = strconv.Atoi(y_char[0])
			if err != nil {
				return []float64{}, 0, err
			}
			n.J, err = strconv.Atoi(y_char[1])
			if err != nil {
				return []float64{}, 0, err
			}
			_f, err = strconv.ParseFloat(y_char[2], 32)
			if err != nil {
				return []float64{}, 0, err
			}
			n.V = float32(_f)
			x = append(x, n)
		}
		y_bar := ffm_predict(x, *c.ModelData)
		f_out = append(f_out, y_bar)
		if 1.0 == y {
			loss -= math.Log(y_bar)
		} else {
			loss -= math.Log(1 - y_bar)
		}
	}
	loss /= float64(len(ffm_predict_data))
	return f_out, loss, err
}
