#pragma once
#include "Common.h"
#include "Utils.h"

namespace redmax {

class Joint;

class Actuator {
public:
    std::string _name;
    int _ndof;                      // number of action dofs
    std::vector<int> _index;        // the indices in the control vector u
    Joint* _joint;                  // the joint to apply the motor force.
    VectorX _u;                     // control signals [-1, 1]
    VectorX _ctrl_min, _ctrl_max;   // control range min / max
    VectorX _dofs, _dofs_vel;       // the value and velocity of the actuated dofs
    VectorX _fr;                    // joint force

    Actuator(std::string name, int ndof, VectorX ctrl_min, VectorX ctrl_max);
    Actuator(std::string name, int ndof, dtype ctrl_min, dtype ctrl_max);

    void get_ctrl_range(VectorX& ctrl_min, VectorX& ctrl_max);

    void set_u(const VectorX& u);

    virtual void update_states(const VectorX& dofs, const VectorX& dofs_vel) = 0;
    
    virtual void computeForce(VectorX& fm, VectorX& fr) = 0;
    virtual void computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr) = 0;

    virtual void compute_dfdu(MatrixX& dfm_du, MatrixX& dfr_du) = 0;
    virtual void compute_extra_derivatives(MatrixX& dfm_dqprev, MatrixX& dfm_dqdotprev, MatrixX& dfr_dqprev, MatrixX& dfr_dqdotprev) = 0;
};

}
