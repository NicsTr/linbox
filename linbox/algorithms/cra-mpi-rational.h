/* Copyright (C) 2007 LinBox
 *
 * ========LICENCE========
 * This file is part of the library LinBox.
 *
 * LinBox is free software: you can redistribute it and/or modify
 * it under the terms of the  GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 * ========LICENCE========
 */

#pragma once

namespace LinBox {
    struct IntegerHash {
        size_t operator()(const Integer& n) const
        {
            size_t hash = 0u;
            auto mpz = n.get_mpz();
            auto size = std::abs(mpz->_mp_size);
            for (auto i = 0; i < size; ++i) {
                hash += mpz->_mp_d[i];
            }
            return hash;
        }
    };

    template <class RatCRABase>
    struct MPIRationalRemainder {
        using Domain = typename RatCRABase::Domain;
        using DomainElement = typename RatCRABase::DomainElement;

    protected:
        RatCRABase _builder;
        Communicator* _communicator = nullptr;
        unsigned int _processCount = 0u;

    public:
        template <class Param>
        MPIRationalRemainder(const Param& b, Communicator* c)
            : _builder(b)
            , _communicator(c)
            , _processCount(c->size())
        {
        }

        // @todo Missing non-vector version of operator().
        // But should surely be invisible for us.
        // We should just template BlasVector<Givaro::ZRing<Integer>>

        template <class Function, class PrimeIterator>
        BlasVector<Givaro::ZRing<Integer>>& operator()(BlasVector<Givaro::ZRing<Integer>>& num, Integer& den, Function& Iteration,
                                                       PrimeIterator& primeg)
        {
            // @fixme The prime generator should somehow be used for the MaskPrimeGenerator, right?

            // Check if just sequential
            if (_communicator == nullptr || _processCount == 1) {
                RationalRemainder<RatCRABase> sequential(_builder);
                return sequential(num, den, Iteration, primeg);
            }

            if (_communicator->master()) {
                master_task(Iteration);
                return _builder.result(num, den);
            }
            else {
                worker_task(Iteration);

                // Dummy return, to prevent warnings
                return num;
            }
        }

        template <class Function>
        void worker_task(Function& Iteration)
        {
            LinBox::MaskedPrimeIterator<LinBox::IteratorCategories::HeuristicTag> gen(_communicator->rank(), _processCount);
            std::unordered_set<Integer, IntegerHash> usedPrimes;

            BlasVector<Domain> result;

            while (true) {
                // Stop when poison pill is 1
                int poisonPill = 0;
                _communicator->recv(poisonPill, 0);
                if (poisonPill == 1) break;

                // Find the next valid prime
                Integer p = *(++gen);
                while (_builder.noncoprime(p) || usedPrimes.find(p) != usedPrimes.end()) {
                    p = *(++gen);
                };
                usedPrimes.insert(p);

                // Compute mod p
                Domain D(p);
                Iteration(result, D);

                // Send back the prime and the result to master
                _communicator->send(p, 0);
                _communicator->send(result, 0);
            }
        }

        template <class Function>
        void master_task(Function& Iteration)
        {
            // ----- Init

            int startWork = 0;
            for (auto i = 1u; i < _processCount; i++) {
                _communicator->send(startWork, i);
            }

            // @note We need a valid domain to initialize the builder,
            // however, we don't know any prime yet.
            // Fact is it doesn't really matter here.
            Domain D(3);
            BlasVector<Domain> result;
            _builder.initialize(D, Iteration(result, D));

            // ----- Compute (CRA reconstruction according to workers' results)

            int poisonPillsLeft = _processCount - 1;
            while (poisonPillsLeft > 0) {
                Integer p;
                _communicator->recv(p, MPI_ANY_SOURCE);
                int process = (_communicator->status()).MPI_SOURCE;
                _communicator->recv(result, process);

                // Send the poison pill, or continue
                int poisonPill = _builder.terminated() ? 1 : 0;
                _communicator->send(poisonPill, process);
                if (poisonPill == 1) {
                    poisonPillsLeft -= 1;
                    continue;
                }

                Domain D(p);
                _builder.progress(D, result);
            }
        }
    };
}
