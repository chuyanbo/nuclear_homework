# nuclear_homework
## 温家学派核燃料计算器

## ![项目图](./pic/毁灭菇.png)
### 提供了三种解析解法，对角化，BDF和解ODE




数据库结构：
```
{
"chain":
    [
        "chain_1":{
            "nuclei": # 反应链中所有核素
                [
                    nuclei_name_1,
                    nuclei_name_2,
                    ...
                ],
            "reaction": # 所有反应道
                [
                    {            
                        "origin_nuclei_name":dtype=string, # 反应物核素名称 e.g. U_238

                        "decay_status":dtype=bool, # 是否衰变，如果衰变则为1
                        "decay_constant":dtype=float64, # 衰变常数

                        "absorb_status":dtype=string, # 是否n,gamma吸收，如果吸收则为1
                        "absorb_gamma_cross_section":dtype=float64, # n,gamma吸收截面
                        "absorb_fission_cross_section":dtype=float64, # n,fission吸收截面
                        "non_linear_factor":(json,dtype=string) # 非线性因子
                            {
                                [
                                    [temperature,g_factor], # 温度：非线性因子
                                    ...
                                ]
                            },
                        
                        "target_nuclei_name":dtype=string:, # n,gamma目标核素
                    },
                    ...
                ]
            },
        ...
    ]
}
```
